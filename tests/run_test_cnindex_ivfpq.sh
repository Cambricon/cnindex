#bash

TEST_SEARCH_RANDOM_DATASET=1
TEST_SEARCH_REAL_DATASET=0
TEST_ADD=0
TEST_REMOVE=0
TEST_MULTI_INSTANCES=0
TEST_CONCURRENCY=0

#./test_cnindex_ivfpq device_id mode nq d M nbits ntotal nlist nprobe topK
#         mode: s: search test using random dataset
#               "test-data": search test using dataset from files
#               a: add test
#               r: remove test
#               m: multi instances search test
#               c: concurrency test

if [ $TEST_SEARCH_RANDOM_DATASET == 1 ]; then
  for ntotal in 50000000 100000000
  do
    for nq in 1 10 20 30 40 50
    do
      ./test_cnindex_ivfpq 0 s $nq 256 32 8 $ntotal 1024 1024 1000
    done
  done
fi

if [ $TEST_SEARCH_REAL_DATASET == 1 ]; then
  # 2M dataset
  if [ ! -d ./test_data/ivfpq-data-2M ]; then
    if [ -f ./test_data/ivfpq-data-2M.tar.gz ]; then
      tar xf ./test_data/ivfpq-data-2M.tar.gz -C ./test_data
    fi
    DATA_DIR=0
  else
    DATA_DIR=1
  fi
  if [ -d ./test_data/ivfpq-data-2M ]; then
    ./test_cnindex_ivfpq 0 ./test_data/ivfpq-data-2M 32 256 32 8 2000000 1024 128 32
  fi
  if [ $DATA_DIR != 1 ]; then
    rm -rf test_data/ivfpq-data-2M
  fi
  # 10M dataset
  if [ ! -d ./test_data/ivfpq-data-10M ]; then
    if [ -f ./test_data/ivfpq-data-10M.tar.gz ]; then
      tar xf ./test_data/ivfpq-data-10M.tar.gz -C ./test_data
    fi
    DATA_DIR=0
  else
    DATA_DIR=1
  fi
  if [ -d ./test_data/ivfpq-data-10M ]; then
    ./test_cnindex_ivfpq 0 ./test_data/ivfpq-data-10M 32 256 32 8 10000000 1024 128 32
  fi
  if [ $DATA_DIR != 1 ]; then
    rm -rf test_data/ivfpq-data-10M
  fi
fi

if [ $TEST_ADD == 1 ]; then
  for dim in 256 512 1024
  do
    for m in 32 64
    do
      for nadd in 10000 100000 1000000
      do
        for nbatch in 16 32
        do
          ./test_cnindex_ivfpq 0 a $nadd $dim $m 8 $nadd 1024 128 $nbatch
        done
      done
    done
  done
fi

if [ $TEST_REMOVE == 1 ]; then
  for ntotal in 10000 100000 1000000
  do
    ./test_cnindex_ivfpq 0 r 10000 256 32 8 $ntotal 1024 128
  done
fi

if [ $TEST_MULTI_INSTANCES == 1 ]; then
  ./test_cnindex_ivfpq 0 m 16 256 32 8 2000000 1024 128
fi

if [ $TEST_CONCURRENCY == 1 ]; then
  ./test_cnindex_ivfpq 0 c
fi
