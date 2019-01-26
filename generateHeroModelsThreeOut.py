# Implementation of a simple MLP network with one hidden layer. Tested on the iris data set.
# NOTE: In order to make the code simple, we rewrite x * W_1 + b_1 = x' * W_1'
# where x' = [x | 1] and W_1' is the matrix W_1 appended with a new row with elements b_1's.
# Similarly, for h * W_2 + b_2
import tensorflow as tf
import numpy as np
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt

datadir = "/data"
listofFiles = listdir(datadir)
listofFiles.sort()
onlyfiles = [f for f in listofFiles if isfile(join(datadir, f))]

inputData = []
count = 0
heroCount = 0

heroes = []
for file in onlyfiles:
    if ".dat" in file and not ".data" in file:
        with open(datadir + file, 'r') as f:
            if not ("skill.dat" in file) and not "skillNormal.dat" in file:
                print(file)
                content = f.readlines()
                inputData.append(np.zeros(shape=[len(content), len(content[0].split())]))
                heroCount = heroCount + 1
                heroes.append(file[0:-4])
                count = 0
                for x in content:

                    # print("len is " + str(len(x.split())))
                    # print("content is " + str(x.split()))
                    if len(x.split()) != 0:
                        inputData[heroCount - 1][count] = (np.asarray([(float(x)) for x in x.split()]))
                        count = count + 1

                if heroCount > 200:
                    break
                    # print("count is " + str(count))
labels = []
with open(datadir + "skillNormal.dat", 'r') as f:
    content = f.readlines()
    labels.append(np.zeros(shape=[len(content), 3*len(content[0].split())]))
    count = 0
    for x in content:
        # print("count is " + str(count))
        labels[0][count] = (np.asarray([float(x),float(x),float(x)]))
        count = count + 1
weightsRaw = []
heroCount = 0
for file in onlyfiles:
    if ".weight" in file:
        print(file)
        with open(datadir + file, 'r') as f:
            print(file)
            content = f.readlines()
            weightsRaw.append(np.zeros(shape=[len(content), len(content[0].split())]))
            heroCount = heroCount + 1
            count = 0
            for x in content:
                # if count < 1000:
                #     print("len is " + str(len(x.split())))
                #     print("content is " + str(x))
                #     print("storing in " + str(heroCount - 1) + " " + str(count))
                weightsRaw[heroCount - 1][count] = (np.asarray(float(x)*float(x)*float(x)*float(x)))
                count = count + 1
            if heroCount > 200:
                break
labelsPercentiled = []


# RANDOM_SEED = 42
# tf.set_random_seed(RANDOM_SEED)


def init_weights(shape, wName):
    """ Weight initialization """
    weights = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(weights, name=wName)


# relu_maxout -> relu dropout -> sigmoid, 400,200,400,2
# Epoch = 438, train accuracy = 7.68%, test accuracy = 7.62%
# Epoch = 438, train accuracy = 7.68%, test accuracy = 7.62% squared = 0.00581%
# relu maxout batchnorm -> tanh dropout -> sigmoid, 400,200,400,2
# was around 7.75
# relu maxout -> tanh dropout -> sigmoid, 400,200,400,2
# no difference from above
# relu maxout batchnorm(fixed?) -> tanh dropout -> sigmoid, 400,200,400,2
# certainly noisier, but no different if not worse
# relu -> relumaxout
# Epoch = 438, train accuracy = 7.69%, test accuracy = 7.64%
# Epoch = 438, train accuracy = 7.68%, test accuracy = 7.67% squared = 0.00588%
# same as abovem 100 neurons only
def forwardprop(X, w_1, w_2, w_3,b_1,b_2,b_3, keep_prop, batch_size=128):
    """
    Forward-propagation.
    IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
    """
    # resultMultiplier = tf.floor(tf.add(tf.reduce_mean(X,axis=1),1.0))
    # X=tf.layers.batch_normalization(X,training=tf.less(keep_prop,tf.constant(1.0)))
    h1 = tf.nn.relu(tf.add(b_1,tf.matmul(X, w_1)))  # The \sigma function
    # h1 = maxout.max_out(h1,200)
    # h1 = tf.layers.batch_normalization(h1,training=(keep_prop!=1.0))
    # h1 = tf.nn.batch_normalization(h1,0,2,0.0,1.0,.0001,)
    # .layer(1,new BatchNormalization.Builder().nIn(numHidden).nOut(numHidden).gamma(2).beta(0.001).decay(.99).updater(Updater.SGD).build())

    # h1 = tf.layers.batch_normalization(h1, training=tf.less(keep_prop,tf.constant(1.0)))
    h2 = tf.nn.relu(tf.add(b_2,tf.matmul(h1, w_2)))
    # print("blah :" + str( keep_prop))
    h2 = tf.nn.dropout(h2,keep_prop)
    # h2 = maxout.max_out(h2, 50)
    # h2 = maxout.max_out(h2, 200)
    # h2 = tf.nn.dropout(h2, keep_prob=keep_prop)

    yhat = tf.add(b_3,tf.matmul(h2, w_3))  # The \varphi function
    # yhat=tf.layers.batch_normalization(yhat, training=tf.less(keep_prop,tf.constant(1.0)))
    yhat2 = tf.nn.sigmoid(yhat, name="prediction")

    # averages = tf.reduce_mean(X, axis=0)
    # results = tf.zeros(shape=[batch_size, 1])
    #
    # # scatter = None
    #
    # for k in range(batch_size):
    #     # tf.gather(results,[k]) = tf.cond(tf.gather(averages,[k]))
    #     indices = tf.constant(k, shape=[1, 1],dtype="int32")
    #     updateValue =  tf.cond(tf.less(tf.gather(averages, tf.constant(k, dtype="int32")),tf.constant(0.0)),
    #                                 lambda: tf.constant(1),#tf.gather(tf.gather(averages, tf.constant(k, shape=[1, 1],dtype="int32")),[0,0]),
    #                                 lambda: tf.constant(0))#tf.gather(tf.gather(yhat2, tf.constant([tf.constant(k,dtype="int32"),tf.constant(0,dtype="int32")],dtype="int32"))),[0,0])
    #
    #     updates = updateValue
    #     shape = tf.shape(results)
    #     if scatter == None:
    #         scatter = tf.scatter_nd(indices, updates, shape)
    #     else:
    #         scatter = tf.group([scatter,tf.scatter_nd(indices, updates, shape)],1)
    # print("building shape is " + str(tf.Session().run(yhat2, feed_dict={X:np.zeros([2,18])})))
    return yhat2


def get_batch(batch_size, maxIndex, trainX, trainY, trainWeights):
    bX = np.zeros(shape=(batch_size, len(trainX[0])))
    bY = np.zeros(shape=(batch_size, len(trainY[0])))
    bW = np.zeros(shape=(batch_size, len(trainWeights[0])))
    #  bX[0] = 0
    bX[0] = trainX[0]
    validSampleCount = 0
    while validSampleCount < batch_size:
        randIndex = np.random.randint(0, maxIndex)
        # print("train weights  " + str(trainWeights[randIndex][0]))
        if trainWeights[randIndex][0] != 0.0:
            bX[validSampleCount] = trainX[randIndex]
            bY[validSampleCount] = trainY[randIndex]
            bW[validSampleCount] = trainWeights[randIndex]
            validSampleCount = validSampleCount + 1
    return bX, bY, bW


recordPerformance = []


def main(heroIndex):
    print(tf.Session().run(tf.reduce_mean([[1.0, 2, 3], [4.0 , 5, 6]], axis=0)))  # Tensor("Mean_1:0", shape=(3,), dtype=int32)
    print(tf.Session().run(tf.reduce_mean([[1.0, 2, 3], [4.0, 5, 6]], axis=1)))  # Tensor("Mean_2:0", shape=(2,), dtype=int32)
    print(tf.Session().run(tf.transpose(tf.multiply(tf.transpose([[1.0], [2], [3]]), [4.0, 5, 6]))))  # Tensor("Mean_2:0", shape=(2,), dtype=int32)
    # train_X, test_X, train_y, test_y = get_iris_data()
    with tf.variable_scope(heroes[heroIndex]):
        currentLearnRate = .12
        train_X = inputData[heroIndex]
        test_X = inputData[heroIndex]
        train_y = labels[0]
        # labelsPercentiled
        test_y = labels[0]
        heroWeights = weightsRaw[heroIndex]
        # Layer's sizes
        print("shape of hero Weights ", heroWeights.shape)
        # print("shape is " + str(train_X.shape))
        # print("len is " + str(train_X.shape[1]))

        x_size = train_X.shape[1]  # Number of input nodes: 4 features and 1 bias
        h_size = 100  # Number of hidden nodes
        y_size = 3  # Number of outcomes (3 iris flowers)

        # Symbols
        X = tf.placeholder("float", shape=[None, x_size], name="X")
        y = tf.placeholder("float", shape=[None, y_size], name="y")
        keep_prob = tf.placeholder("float", name="keep_prob")
        learn_rate = tf.placeholder("float", name="learn_rate")

        weight = tf.placeholder("float", shape=[None, 1], name="weight")
        # Weight initializations
        w_1 = init_weights((x_size, h_size), heroes[heroIndex] + "w1")
        w_2 = init_weights((h_size, h_size // 2), heroes[heroIndex] + "w2")
        w_3 = init_weights((h_size // 2, y_size), heroes[heroIndex] + "w3")
        b_1 = init_weights([h_size],heroes[heroIndex]+"b1")
        b_2 = init_weights([h_size//2], heroes[heroIndex] + "b2")
        b_3 = init_weights([y_size], heroes[heroIndex] + "b3")
        # Forward propagation
        yhat = forwardprop(X, w_1, w_2, w_3,b_1,b_2,b_3,keep_prob)
        predict = yhat

        # Backward propagation
        cost = tf.multiply(tf.losses.absolute_difference(y, yhat), weight, name="cost")
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            updates = tf.train.GradientDescentOptimizer(learn_rate, name="optimizer").minimize(cost)
        # updates = tf.train.AdamOptimizer( name="optimizer").minimize(cost)
        # Run SGD

        sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=8))
        init = tf.global_variables_initializer()

        sess.run(init)
        saver = tf.train.Saver()
        maxIndex = int(len(train_X) * .9)
        endIndex = len(train_X)
        plt.axis([0, 10, 0, .25])
        plt.ion()
        plt.clf()
        minErr = .3
        maxErr = .112
        batch_size = 128
        train_accuracy = 0
        test_accuracy = 0
        for epoch in range(200):
            # Train with each example
            # print("weights: " + str(weights[0][0:1000]))
            # if epoch == 100:
            #     currentLearnRate = 0.02
            # if epoch == 500:
            #     currentLearnRate = 0.01
            # if epoch == 100000:
            #     updates = tf.train.GradientDescentOptimizer(0.0005).minimize(cost)
                # config=tf.ConfigProto(
                #     intra_op_parallelism_threads=NUM_THREADS)
                # likely? 60 neurons, dropout, with batch=1
                # Epoch = 844, train accuracy = 8.17%, test accuracy = 8.03%
                # Epoch = 844, train accuracy = 8.18%, test accuracy = 8.04% squared = 0.00647%
                # 400 neurons, no dropout, no batch
                # Epoch = 919, train accuracy = 7.94%, test accuracy = 7.78%
                # Epoch = 919, train accuracy = 7.94%, test accuracy = 7.76% squared = 0.00603%
            # 400 neurons with dropoout
            # Epoch = 1447, train accuracy = 8.00%, test accuracy = 7.82%
            # Epoch = 1447, train accuracy = 7.84%, test accuracy = 7.81% squared = 0.00609%
            # i also only did .1 until iter 200 instead of 500, so  that might make the difference here
            if epoch %10 == 0:
                print ("epoch: " + str(epoch))
            # print("shapw of hero weights " + str(heroWeights[0:1000].shape))
            # print("shapw of train_y " + str(train_y[0:1000].shape))
            # print("shape of predict is: " + str(sess.run(tf.shape(predict),
            #                                     feed_dict={keep_prob: 1.0, X: train_X[0:1000], y: train_y[0:1000],
            #                                                weight: heroWeights[0:1000]})))
            # train_accuracy = np.average(
            #     np.abs(train_y[0:1000] - sess.run(predict,
            #                                       feed_dict={keep_prob: 1.0, X: train_X[0:1000], y: train_y[0:1000],
            #                                                  weight: heroWeights[0:1000]})),
            #     weights=weightsRaw[heroIndex][0:1000])
            #
            # test_accuracy = np.average(
            #     np.abs(test_y[maxIndex:endIndex] - sess.run(predict, feed_dict={keep_prob: 1.0,
            #                                                                     X: test_X[maxIndex:endIndex],
            #                                                                     y: test_y[maxIndex:endIndex],
            #                                                                     weight: heroWeights[
            #                                                                             maxIndex:endIndex]})),
            #     weights=weightsRaw[heroIndex][maxIndex:endIndex])

            # print(("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%% " + heroes[heroIndex])
            #       % (epoch + 1, 100. * train_accuracy, 100. * test_accuracy))

            total_parameters = 0
            for variable in tf.trainable_variables():
                # shape is an array of tf.Dimension
                shape = variable.get_shape()
                # print(shape)
                # print(len(shape))
                variable_parametes = 1
                for dim in shape:
                    # print(dim)
                    variable_parametes *= dim.value
                # print(variable_parametes)
                total_parameters += variable_parametes
            # print(total_parameters)

            for i in range(200):
                #            print("shape2 is " + str(train_X[i: i + 1].shape))
                #           print("shape3 is " + str(train_y[i: i + 1].shape))
                randIndex = np.random.randint(0, maxIndex)
                # if train_X[randIndex][0] != -1.0:
                batchX, batchY, batchWeights = get_batch(batch_size, maxIndex, train_X, train_y, heroWeights)
                if len(batchX) != 128:
                    print("size of batch: " + str(len(batchX)))
                sess.run(updates,
                         feed_dict={learn_rate: currentLearnRate, keep_prob: .5, X: batchX,
                                    y: batchY,
                                    weight: batchWeights})

            # train_accuracy = np.average(
            #      np.abs(train_y[0:maxIndex] - sess.run(didict,
            #      feed_dict={keep_prob: 1.0, X: train_X[0:maxIndex], y: train_y[0:maxIndex],
            #      weight: heroWeights[0:maxIndex]})),
            # weights=weightsRaw[heroIndex][0:maxIndex])
            # test_accuracy = np.average(
            # np.abs(test_y[maxIndex:endIndex] - sess.run(predict, feed_dict={keep_prob: 1.0,
            # X: test_X[maxIndex:endIndex],
            # y: test_y[maxIndex:endIndex],
            # weight: heroWeights[maxIndex:endIndex]})),
            # weights=weightsRaw[heroIndex][maxIndex:endIndex])

            # print(("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%% squared = %.5f%%" + heroes[heroIndex])
            #     % (epoch + 1, 5000. * train_accuracy, 5000. * test_accuracy, test_accuracy * test_accuracy))
            # if epoch % 5 == 0:
            # plt.scatter(epoch, train_accuracy, color='r')
            # plt.scatter(epoch, test_accuracy,  color='b')
            # if train_accuracy < minErr:
            #     minErr = train_accuracy

            # if test_accuracy < minErr:
            #     minErr = test_accuracy

            # plt.axis([0, epoch+1, minErr, maxErr])
            # plt.pause(0.05)

    recordPerformance.append(
        ("REPORT: train accuracy = %.2f%%, test accuracy = %.2f%% squared = %.5f%%" + heroes[heroIndex])
        % (100. * train_accuracy, 100. * test_accuracy, test_accuracy * test_accuracy))
    for s in recordPerformance:
        print(s)
    saver.save(sess, datadir + "tensorModels/" + heroes[heroIndex] + "season9.5", global_step=2000)
    tf.reset_default_graph()
    sess.close()


if __name__ == '__main__':
    for i in range(len(heroes)):
        main(i)
