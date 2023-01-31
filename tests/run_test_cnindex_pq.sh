#bash

TEST_SEARCH_RANDOM_DATASET=1
TEST_ADD=0
TEST_REMOVE=0

#./test_cnindex_pq device_id mode nq d M nbits ntotal topK 
#         mode: s: search test using random dataset
#               "test-data": search test using dataset from files
#               a: add test
#               r: remove test

if [ $TEST_SEARCH_RANDOM_DATASET == 1 ]; then
  for ntotal in 50000000 100000000
  do
    for nq in 1 10 20 30 40 50
    do
      ./test_cnindex_pq 0 s $nq 256 32 8 $ntotal 1000
    done
  done
fi

if [ $TEST_ADD == 1 ]; then
  for dim in 256 512 1024
  do
    for m in 32 64
    do
      for nadd in 10000 100000 1000000
      do
        ./test_cnindex_pq 0 a $nadd $dim $m 8 $nadd
      done
    done
  done
fi

if [ $TEST_REMOVE == 1 ]; then
  for ntotal in 10000 100000 1000000
  do
    ./test_cnindex_pq 0 r 10000 256 32 8 $ntotal
  done
fi
