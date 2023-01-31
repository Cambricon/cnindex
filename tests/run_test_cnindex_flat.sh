#bash

TEST_SEARCH_RANDOM_DATASET=1
TEST_ADD=0
TEST_REMOVE=0

#./test_cnindex_flat device_id mode nq d ntotal topK metric
#         mode: s: search test using random dataset
#               "test-data": search test using dataset from files
#               a: add test
#               r: remove test

if [ $TEST_SEARCH_RANDOM_DATASET == 1 ]; then
  for ntotal in 131072 1048576 10485760
  do
    for topk in 1 3 100 1000
    do
      for nq in 1 100 10000
      do
        ./test_cnindex_flat 0 s $nq 256 $ntotal $topk 1
      done
    done
  done
fi

if [ $TEST_ADD == 1 ]; then
  for ntotal in 2000000
  do
    for nadd in 10000 100000 1000000
    do
      ./test_cnindex_flat 0 a $nadd 256 $ntotal
    done
  done
fi

if [ $TEST_REMOVE == 1 ]; then
  for ntotal in 10000 100000 1000000
  do
    for nremove in 10000
    do
      ./test_cnindex_flat 0 r $nremove 256 $ntotal
    done
  done
fi
