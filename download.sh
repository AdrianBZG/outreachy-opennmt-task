#!/bin/bash

wget -P data https://s3.amazonaws.com/opennmt-trainingdata/toy-ende.tar.gz
wget -P data http://data.statmt.org/wmt17/translation-task/rapid2016.tgz
wget -P data http://www.statmt.org/wmt15/wiki-titles.tgz

cd data
tar -xzf rapid2016.tgz
tar -xzf toy-ende.tar.gz
tar -xzf wiki-titles.tgz
