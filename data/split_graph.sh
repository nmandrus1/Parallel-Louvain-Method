#!/bin/sh

L= wc -l $1 | cut -d' ' -f1
N= $((L/$2 + 1))
echo $N
split -d -l $N  $1

