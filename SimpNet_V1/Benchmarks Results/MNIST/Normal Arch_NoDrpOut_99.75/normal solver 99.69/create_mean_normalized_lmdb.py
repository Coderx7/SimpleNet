import caffe
import lmdb
import numpy as np
from caffe.proto import caffe_pb2
import cv2
import sys

db_train = lmdb.open('mnist_train_lmdb')
db_train_txn = db_train.begin(write=True)
db_test = lmdb.open('mnist_test_lmdb')
db_test_txn = db_test.begin(write=True)

datum = caffe_pb2.Datum()

index = sys.argv[0]
size_train = 60000
size_test = 10000
data_train = np.zeros((size_train, 1, 28, 28))
label_train = np.zeros(size_train, dtype=int)

data_test = np.zeros((size_test, 1, 28, 28))
label_test = np.zeros(size_test, dtype=int)

print 'Reading training data...'
i = -1
for key, value in db_train_txn.cursor():
    i = i + 1
    if i % 1000 == 0:
        print i
    if i == size_train:
        break
    datum.ParseFromString(value)
    label = datum.label
    data = caffe.io.datum_to_array(datum)
    data_train[i] = data
    label_train[i] = label


print 'Reading test data...'
i = -1
for key, value in db_test_txn.cursor():
    i = i + 1
    if i % 1000 == 0:
        print i
    if i ==size_test:
        break
    datum.ParseFromString(value)
    label = datum.label
    data = caffe.io.datum_to_array(datum)
    data_test[i] = data
    label_test[i] = label


print 'Computing statistics...'
mean = np.mean(data_train)
std = np.std(data_train)

print mean
print std

print mean.shape
print std.shape

#np.savetxt('mean_mnist.txt', mean)
#np.savetxt('std_mnist.txt', std)

print 'Normalizing'
data_train = data_train - mean
data_train = data_train/std
data_test = data_test - mean
data_test = data_test/std

#Zero Padding
#print 'Padding...'
#npad = ((0,0), (0,0), (4,4), (4,4))
#data_train = np.pad(data_train, pad_width=npad, mode='constant', constant_values=0)
#data_test = np.pad(data_test, pad_width=npad, mode='constant', constant_values=0)

print 'Outputting training data'
lmdb_file ='mnist_train_lmdb_norm2'
batch_size = size_train

db = lmdb.open(lmdb_file, map_size=int(data_train.nbytes))
batch = db.begin(write=True)
datum = caffe_pb2.Datum()

for i in range(size_train):
    if i % 1000 == 0:
        print i

    # save in datum
    datum = caffe.io.array_to_datum(data_train[i], label_train[i])
    keystr = '{:0>5d}'.format(i)
    batch.put( keystr, datum.SerializeToString() )

    # write batch
    if(i + 1) % batch_size == 0:
        batch.commit()
        batch=db.begin(write=True)
        print (i + 1)

# write last batch
if (i+1) % batch_size != 0:
    batch.commit()
    print 'last batch'
    print (i + 1)

print 'Outputting test data'
lmdb_file = 'mnist_test_lmdb_norm2'
batch_size = size_test

db = lmdb.open(lmdb_file,map_size=int(data_test.nbytes))
batch = db.begin(write=True)
datum = caffe_pb2.Datum()

for i in range(size_test):
    # save in datum
    datum = caffe.io.array_to_datum(data_test[i], label_test[i])
    keystr = '{:0>5d}'.format(i)
    batch.put( keystr, datum.SerializeToString() )

    # write batch
    if(i + 1) % batch_size == 0:
        batch.commit()
        batch = db.begin(write=True)
        print (i + 1)

# write last batch
if (i+1) % batch_size != 0:
    batch.commit()
    print 'last batch'
    print (i + 1)
