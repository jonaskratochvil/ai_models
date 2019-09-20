#!/bin/bash
DATA_ROOT=$1

name=Dvacet

pushd $DATA_ROOT
      for t in test train dev ; do
           ln -s $name/$t
      done
      ln -s $name/arpa_bigram arpa-bigram
 popd
