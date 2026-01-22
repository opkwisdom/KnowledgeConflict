#!/bin/bash

declare -A scores
scores["tom"]=85
scores["sally"]=98

for key in "${!scores[@]}"; do
	echo "data/$key/retrieved/${scores[$key]}"
done
