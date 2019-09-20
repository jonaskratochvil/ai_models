#!/bin/bash
file=$1
rsync -avz /home/jonas/Jonas-zaloha/Translate_vypisky/Ceskyrozhlas/Poslanecka_Snemovna/files/${file} \
 jkratochvil@tap.ms.mff.cuni.cz:/lnet/ms/data/cesky-rozhlas-prepisy/data/rozhlas_data/new_data/train/ || exit 1;

rm -r $file
