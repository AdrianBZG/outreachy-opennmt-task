#!/bin/bash

wget -P data https://s3.amazonaws.com/opennmt-trainingdata/toy-ende.tar.gz
wget -P data http://data.statmt.org/wmt17/translation-task/rapid2016.tgz

cd data
tar -xzf rapid2016.tgz
tar -xzf toy-ende.tar.gz
