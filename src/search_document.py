from transformers.file_utils import PaddingStrategy
from transformers import (
    AutoTokenizer,
    AutoModel,
    PreTrainedTokenizerFast,
    DataCollatorWithPadding,
    HfArgumentParser,
    BatchEncoding
)
from omegaconf import OmegaConf, DictConfig
from pyserini.search.faiss import FaissSearcher
from datasets import load_dataset
import torch
import json
import os
from torch.utils.data import DataLoader
from functools import partial
from tqdm import tqdm
from dataclasses import dataclass, field, asdict
import numpy as np
from typing import Tuple, List, Any, Dict, Optional
import faiss


from utils import (
    setup_logger, load_config, load_json_data, move_to_cuda, has_answer
)



DENSE_ENCODER = 'wikipedia-dpr-100w.ance-multi'

args = load_config()


JsonType = Dict[str, Any]

# FaissSearcher return type
@dataclass
class DenseSearchResult:
    docid: str
    score: float

@dataclass
class Context:
    hasanswer: bool
    id: int
    score: float
    text: str
    title: Optional[str] = None


@dataclass
class QAExample:
    question: str
    answers: List[str] = field(default_factory=list)
    num_answer: int = 0
    ctxs: List[Context] = field(default_factory=list)



def _query_transform_func(tokenizer: PreTrainedTokenizerFast,
                             args,
                             examples: Dict[str, str]) -> BatchEncoding:
    question_key = "question"
    if question_key not in examples:
        question_key = "claims" # FactKG
    
    batch_dict = tokenizer(
        text=examples[question_key],
        max_length=args.q_max_len,
        padding=PaddingStrategy.DO_NOT_PAD,
        truncation=True
    )
    return batch_dict


@torch.no_grad()
def _worker_encode_queries(gpu_idx: int, logger) -> Tuple[np.ndarray, List[JsonType]]:
    # Load dataset
    input_path = os.path.join(args.data_dir, args.input_file)
    dataset = load_dataset("json", data_files=input_path)["train"]
    original_dataset = load_json_data(input_path)
    logger.info("Load {} samples from {}".format(len(dataset), input_path))

    # Set model
    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(args.base_model)
    model: AutoModel = AutoModel.from_pretrained(args.base_model)
    model.eval()
    model.cuda()
    
    dataset.set_transform(partial(_query_transform_func, tokenizer, args))
    # dataset = dataset.shard(num_shards=torch.cuda.device_count(),
    #                         index=gpu_idx,
    #                         contiguous=True)
    logger.info('GPU {} needs to process {} examples'.format(gpu_idx, len(dataset)))
    torch.cuda.set_device(gpu_idx)

    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.dataloader_num_workers,
        collate_fn=data_collator,
        pin_memory=True
    )
    
    encoded_embeds = []
    for batch_dict in tqdm(data_loader, desc='query encoding'):
        batch_dict = move_to_cuda(batch_dict)
        
        outputs = model(**batch_dict).pooler_output  # (B, D)
        encoded_embeds.append(outputs.detach().cpu())   # Should be numpy array

    query_embeds = np.concatenate(encoded_embeds, axis=0)     # (N, D)
    
    return query_embeds, original_dataset


@torch.no_grad()
def _worker_batch_search(gpu_idx: int, logger, searcher) -> List[QAExample]:
    query_embeds, original_dataset = _worker_encode_queries(gpu_idx, logger)
    query_ids = [ex['id'] for ex in original_dataset]
    assert len(query_embeds) == len(original_dataset), "Length of query_embeds and original_dataset should be same"
    
    def parse_document(docid) -> Tuple[str, str]:
        doc = searcher.doc(docid)   # 수정됨
        doc = json.loads(doc.raw())
        splitted = doc["contents"].split('\n')
        title, text = splitted[0], "".join(splitted[1:])
        return title, text

    def has_answer(text, answers) -> bool:
        if text is None:
            return False
        if isinstance(answers, dict):
            answers = answers["aliases"]    # TriviaQA
        for ans in answers:
            if ans in text:
                return True
        return False

    all_dataset: List[QAExample] = []
    for i in tqdm(range(0, len(original_dataset), args.search_batch_size), desc='Search with Faiss'):
        batch_embeds = query_embeds[i:i+args.search_batch_size]
        batch_ids = query_ids[i:i+args.search_batch_size]
        batch_dataset = original_dataset[i:i+args.search_batch_size]

        topk_results: Dict[str, DenseSearchResult] = searcher.batch_search(batch_embeds, batch_ids, k=args.search_topk)
        # 원본 배치 내 위치 빠르게 찾기
        pos_in_batch = {ex['id']: idx for idx, ex in enumerate(batch_dataset)}

        for qid, search_res in topk_results.items():
            sample = batch_dataset[pos_in_batch[qid]]
            num_answer = 0
            ctxs: List[Context] = []
            for res in search_res:
                title, text = parse_document(res.docid)

                answers = sample.get("answer", None)
                if answers is None:
                    answers = sample.get("answers", [])  # WebQA
                if answers is None:
                    answers = sample.get("possible_answers", None)  # PopQA
                if answers is None:
                    answers = sample.get("best_answer", None)   # TruthfulQA
                    answers = [answers] if answers is not None else []
                if answers is None:
                    answers = sample.get("entity", None)    # FactKG

                hasanswer = has_answer(text, answers)
                if hasanswer:
                    num_answer += 1
                ctxs.append(
                    Context(
                        hasanswer=hasanswer,
                        id=int(res.docid),
                        score=float(res.score),
                        text=text,
                        title=title
                    )
                )

            question_key = "question"
            if question_key not in sample:
                question_key = "claims" # FactKG

            all_dataset.append(
                QAExample(
                    question=sample[question_key],
                    answers=answers,
                    num_answer=num_answer,
                    ctxs=ctxs
                )
            )
    return all_dataset

def save_to_json(dataset: List[QAExample], output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        for x in dataset:
            f.write(json.dumps(asdict(x)) + '\n')


def get_ivf_index(
        cpu_index,
        logger,
        metric: str = "ip",
        train_size: int = 2_560_000, # 샘플 학습 크기 (최대 100만 권장)
        train_bs: int = 100_000,     # train 벡터 준비 배치 크기
        add_bs: int = 200_000,       # 전체 add 배치 크기
        ivf_filename: str = "ivf_flat.faiss"
    ) -> faiss.Index:
    os.makedirs(args.ivf_dir, exist_ok=True)
    ivf_path = os.path.join(args.ivf_dir, ivf_filename)

    # 1) 이미 만들어둔 IVF가 있으면 바로 로드
    if os.path.isfile(ivf_path):
        logger.info(f"[IVF] Loading existing IVF index from: {ivf_path}")
        ivf = faiss.read_index(ivf_path)
        # nprobe 설정(로드 직후는 1일 수 있음)
        if hasattr(ivf, "nprobe"):
            ivf.nprobe = args.nprobe
        logger.info(f"[IVF] Loaded. is_trained={getattr(ivf,'is_trained',None)}, ntotal={ivf.ntotal}, nprobe={getattr(ivf,'nprobe',None)}")
        return ivf

    # 2) 새로 만들기
    d = cpu_index.d
    ntotal = cpu_index.ntotal
    logger.info(f"[IVF] Building new IVF-Flat from CPU index: d={d}, ntotal={ntotal}, metric={metric}, nlist={args.nlist}, nprobe={args.nprobe}")

    faiss_metric = faiss.METRIC_INNER_PRODUCT if metric.lower() == "ip" else faiss.METRIC_L2
    quantizer = faiss.IndexFlatIP(d) if faiss_metric == faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(d)
    ivf = faiss.IndexIVFFlat(quantizer, d, args.nlist, faiss_metric)

    # 2-1) 학습용 샘플 준비 (연속 구간으로 안전하게)
    tsz = int(min(ntotal, max(100_000, train_size)))  # 최소 10만
    logger.info(f"[IVF] Preparing {tsz} training vectors in chunks of {train_bs} ...")
    xt = np.empty((tsz, d), dtype="float32")
    pos = 0
    while pos < tsz:
        cnt = min(train_bs, tsz - pos)
        xt[pos:pos+cnt] = cpu_index.reconstruct_n(pos, cnt)
        pos += cnt
        if pos % (5 * train_bs) == 0 or pos == tsz:
            logger.info(f"[IVF] prepared train vecs: {pos}/{tsz}")

    # 2-2) 학습
    logger.info("[IVF] Training IVF centroids ...")
    ivf.train(xt)
    logger.info(f"[IVF] Training done. is_trained={ivf.is_trained}")

    # 2-3) 전체 벡터 add (청크)
    logger.info(f"[IVF] Adding all {ntotal} vectors in chunks of {add_bs} ...")
    added = 0
    while added < ntotal:
        cnt = min(add_bs, ntotal - added)
        xb = cpu_index.reconstruct_n(added, cnt)
        ivf.add(xb)
        added += cnt
        if added % (5 * add_bs) == 0 or added == ntotal:
            logger.info(f"[IVF] added vectors: {added}/{ntotal}")

    # 2-4) 검색 파라미터 설정 및 저장
    ivf.nprobe = args.nprobe
    faiss.write_index(ivf, ivf_path)
    logger.info(f"[IVF] Saved IVF index to: {ivf_path} (ntotal={ivf.ntotal}, nprobe={ivf.nprobe})")

    return ivf


def retrieve_documents():
    logger = setup_logger("search_document", args.encode_save_dir)
    logger.info("Configuration Loaded:")
    logger.info(OmegaConf.to_yaml(args))

    # CPU 인덱스 로드
    ivf_path = os.path.join(args.ivf_dir, "ivf_flat.faiss")
    searcher = FaissSearcher.from_prebuilt_index(DENSE_ENCODER, None)
    if not os.path.isfile(ivf_path):
        cpu_index = searcher.index
        ivf_index = get_ivf_index(cpu_index, logger)
    else:
        ivf_index = get_ivf_index(None, logger)
    # IVF 인덱스 변경
    searcher.index = ivf_index

    # Dry run만 구현함
    all_dataset = _worker_batch_search(0, logger, searcher)
    logger.info("Done batch search queries")

    output_path = os.path.join(args.encode_save_dir, args.output_path)
    save_to_json(all_dataset, output_path)
    logger.info("Saved to {}".format(output_path))


if __name__ == "__main__":
    retrieve_documents()


# GPU 옵션 설정
# res0 = faiss.StandardGpuResources()
# res0.setTempMemory(512 * 1024 * 1024)  # 512MB만 임시 워크스페이스로 사용
# res1 = faiss.StandardGpuResources()
# res1.setTempMemory(512 * 1024 * 1024)  # 512MB만 임시 워크스페이스로 사용

# # GpuResource 설정
# resources = faiss.GpuResourcesVector()
# resources.push_back(res0)
# resources.push_back(res1)

# # GPU 디바이스 설정
# devices = faiss.Int32Vector()
# devices.push_back(0)
# devices.push_back(1)

# co = faiss.GpuMultipleClonerOptions()
# co.useFloat16 = True  # FP16로 메모리/속도 최적화 (정확도 영향 매우 미미)
# co.shard=True

# # CPU 인덱스를 모든 GPU에 올리기
# gpu_index = faiss.index_cpu_to_gpu_multiple(resources, devices, ivf_index, options=co)
# # gpu_index = faiss.index_cpu_to_all_gpus(cpu_index, co=co, resources=[res, res])
# # GPU 인덱스로 검색기 생성
# searcher = FaissSearcher(None, None)
# searcher.index = gpu_index