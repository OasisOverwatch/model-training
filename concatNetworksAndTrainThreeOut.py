import argparse

import tensorflow as tf
import numpy as np
from os import listdir
from os.path import isfile, join
import Maxout as maxout

datadir = "/home/dfreelan/data/data/"
listofFiles = listdir(datadir)
listofFiles.sort()
onlyfiles = [f for f in  listofFiles if isfile(join(datadir, f))]
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
                if heroCount > 100:
                    break
                    # print("count is " + str(count))
labels = []
with open(datadir + "skillNormal.dat", 'r') as f:
    content = f.readlines()
    labels.append(np.zeros(shape=[len(content), len(content[0].split())]))
    count = 0
    for x in content:
        # print("count is " + str(count))
        labels[0][count] = (np.asarray([float(x)]))
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
                weightsRaw[heroCount - 1][count] = (np.asarray(float(x)))
                count = count + 1
            if heroCount > 100:
                break


# RANDOM_SEED = 42
# tf.set_random_seed(RANDOM_SEED)
sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=64))
def init_bias(shape,wName):
    initializer = tf.constant_initializer(value=0.0, dtype=tf.float32)
    return tf.Variable(initializer(shape=shape), name=wName)


def init_weights(shape, wName):
    """ Weight initialization """
    # weights = tf.random_normal(shape, stddev=0.1)
    # return tf.Variable(weights, name=wName)
    return tf.get_variable(shape=shape, name=wName, initializer=tf.contrib.layers.xavier_initializer())


doMaxOut = True
withTrain = True
modelVars = []
heroInput = []
keepProbs = []
heroLearnAdjustment = {}
scaleDownHeroRate = 1.0
# tf.placeholder("float",
for hero in heroes:
    heroLearnAdjustment[hero] = (tf.placeholder("float", shape=[1]))
trainingEndIndex = len(inputData[0]) *.9
print("training set index is: " + str(trainingEndIndex))

# shape should be....
# [hero][batch_size][inputsize]

def create_all_hero_batch(batch_size, isTrain, labels):
    # heroInput = np.zeros(batch_size)
    # keepProbs = np.zeros(batch_size)
    allHeroBatchData = []
    allHeroBatchLabels = []
    allHeroBatchWeights = np.zeros(shape=[batch_size, len(heroes)])
    if not isTrain:
        allHeroBatchWeights = np.zeros(shape=[len(inputData[0])-int(np.floor(trainingEndIndex)), len(heroes)])
    for hero in range(len(heroes)):
        allHeroBatchData.append([])
        if isTrain:
            allHeroBatchData[hero] = np.zeros(shape=[batch_size, len(inputData[hero][0])])
        else:
            allHeroBatchData[hero] = np.zeros(shape=[len(inputData[0])-int(np.floor(trainingEndIndex)), len(inputData[hero][0])])
    if not isTrain:
        for i in range(len(inputData[0])-int(np.floor(trainingEndIndex))):
            randIndex = int(np.floor(trainingEndIndex))+i
            allHeroBatchLabels.append(labels[0][randIndex])

            for hero in range(len(heroes)):
                allHeroBatchData[hero][i] = inputData[hero][randIndex]
                allHeroBatchWeights[i][hero] = weightsRaw[hero][randIndex]

        return allHeroBatchData, allHeroBatchLabels, allHeroBatchWeights
    for i in range(batch_size):
        randIndex = 0

        if isTrain:
            randIndex = np.random.randint(0, trainingEndIndex)
        else:
            randIndex = np.random.randint(trainingEndIndex, len(inputData[0]))

        allHeroBatchLabels.append(labels[0][randIndex])

        for hero in range(len(heroes)):
            allHeroBatchData[hero][i] = inputData[hero][randIndex]
            allHeroBatchWeights[i][hero] = weightsRaw[hero][randIndex]

    return allHeroBatchData, allHeroBatchLabels, allHeroBatchWeights


def convertToDictionary(batchData, batchLabels, batchWeights, learnRate, weightPlaceholder, y, learnTensor,keepTensor,keepValue):
    myDict = {}
    for i in range(len(heroInput)):
        myDict[heroInput[i]] = batchData[i]
        myDict[keepProbs[i]] = 1.0
    myDict[y] = batchLabels
    myDict[weightPlaceholder] = batchWeights
    myDict[learnTensor] = learnRate
    myDict[keepTensor] = keepValue
    return myDict


def populate_dict(index, y, heroInput, keepProbs):
    dict = {}
    for i in range(len(heroInput)):
        dict[heroInput[i]] = np.reshape(inputData[i][index], (1, len(inputData[i][index])))
        dict[keepProbs[i]] = 1.0
    dict[y] = np.reshape(labels[0][index], [1, 1])
    return dict
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    # saver = tf.train.Saver()
    global secondOutput
    global doMaxOut
    global scaleDownHeroRate
    global withTrain
    global percentTimePlayed
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-o2', '--secondOutput', help='Description for foo argument', required=True, type = str2bool , nargs='?')
    parser.add_argument('-dM', '--doMaxOut', help='Description for bar argument', required=True,  type = str2bool , nargs='?')
    parser.add_argument('-sD', '--scaleDownHeroRate', help='Description for bar argument', required=True, type=float)
    parser.add_argument('-wT', '--withTrain', help='Description for bar argument', required=True,  type = str2bool , nargs='?')
    parser.add_argument('-sS', '--snapShot', help='Description for bar argument', required=True, type=int)

    args = vars(parser.parse_args())
    secondOutput = args['secondOutput']
    doMaxOut = args['doMaxOut']
    saverIndex = args['snapShot']
    scaleDownHeroRate = args['scaleDownHeroRate']
    withTrain = args['withTrain']
    fileName = None
    print("second out: " + str(secondOutput))
    print("ddoMax : " + str(doMaxOut))
    print("values are: " + str(not secondOutput) + " " + str(not doMaxOut) + " " + str(scaleDownHeroRate == 1.0)  + " " + str(withTrain))
    # concatedControlOldLearnNTNormArchitectWITHTRAINFAST at 41400
    if not secondOutput and (not doMaxOut) and scaleDownHeroRate==1.0 and withTrain:
        fileName = "concatedControlOldLearnNTNormArchitectWITHTRAINFAST"
    if secondOutput and not doMaxOut and scaleDownHeroRate==1.0 and withTrain:
        fileName = "concatedControlOldLearnNTNormArchitectWITHTRAINFASTSEecondOutput"
    if not secondOutput and not doMaxOut and scaleDownHeroRate == .1 and withTrain:
        fileName = "concatedControlOldLearnNTNormArchitectWITHTRAIN"
    # concatedControlOldLearnNTExpArchitectWITHTRAIN: 266.394986335
    # 36936
    # 249.63537267
    # test fastLRMaxout: 267.711412895 1200 264.994412072
    # test fastLRMaxout: 246.992130535 2800 241.610342251
    # test fastLRMaxout2Out: 242.229462524 11500 225.887557329
    # test fastLRMaxout: 244.137286386 9086 233.993045182
    if not secondOutput and doMaxOut and scaleDownHeroRate == .1 and withTrain:
        fileName = "concatedControlOldLearnNTExpArchitectWITHTRAIN"
    if not secondOutput and doMaxOut and scaleDownHeroRate == 1.0 and withTrain:
        fileName = "fastLRMaxout2Out"
    if secondOutput and doMaxOut and scaleDownHeroRate == .1 and withTrain:
        fileName = "slowSecondOutMaxOut"
    if secondOutput and doMaxOut and scaleDownHeroRate == 1.0 and withTrain:
        fileName = "fastSecondOutMaxOut"
    if not withTrain:
        fileName = "noTrain"
    fileName = fileName+"_oldRelu"
    # withTrain = True
    print("with train is " + str(withTrain))
    print("file name is " + fileName)
    if fileName==None:
        return
    # fileName = "concatedControlOldLearnNTNormArchitectWITHTRAINFASTSEecondOutput"
    saverDir = datadir + "tensorModels/" + fileName + "-" + str(saverIndex)
    if saverIndex==0:
        saverDir = None
    outputVec = None

    percentTimePlayed = tf.placeholder("float", shape=[None, len(heroes)], name="percentTimePlayed")
    outputVec = concat_networks(outputVec, secondOutput)
    keep_prob = tf.placeholder("float")
    # print("shape before: " +sess.run(tf.shape(outputVec)))
    outputPlusPercent = tf.concat([percentTimePlayed, outputVec], 1)
    
    with tf.variable_scope("local"):
        # multiplier = 2
        # if secondOutput:
        multiplier = 4
        concatWeight = init_weights((27 * multiplier, 400), "concatWeight")
        print("concat weight is initialized right? " + str(concatWeight))
        concatedLayerBiases = init_weights([400], "concatBias")
        concatedLayer = maxout.max_out(
            tf.nn.relu(tf.add(tf.matmul(outputPlusPercent, concatWeight), concatedLayerBiases)), 200)

        midWeights = init_weights((200, 200), "midWeight")
        if doMaxOut:
            midBiases = init_weights([200], "midBiases")
            midLayer = maxout.max_out(tf.nn.relu(tf.add(tf.matmul(concatedLayer, midWeights), midBiases)), 100)
            # midLayerWeights = init_weights((50,50),"midLayerWeig*************hts")


            outWeight = init_weights((200, 200), "outWeight")
            outBiases = init_weights([100], "outBiases")
        else:
            midBiases = init_weights([200], "midBiases")
            midLayer = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(concatedLayer, midWeights), midBiases)),
                                     keep_prob=keep_prob)

            outWeight = init_weights((400, 1), "outWeight")
            outBiases = init_weights([1], "outBiases")

            # print("outWeight: " + str(outWeight))
        lastWeights = init_weights((100, 100), "lastWeigts")
        lastLayer = maxout.max_out(tf.add(tf.matmul(midLayer, lastWeights), outBiases), 50)

        last2Weights = init_weights((50, 50), "last2Weights")
        last2Bias = init_weights([50], "last2Biases")
        last2Layer = maxout.max_out(tf.add(tf.matmul(lastLayer, last2Weights), last2Bias), 25)



        finalBias = init_weights([1], "finalBias")
        finalWeights = init_weights((25, 1), "finalWeights")
        outLayer = tf.nn.sigmoid(tf.add(tf.matmul(last2Layer, finalWeights), finalBias))

        # lastBiases = init_weights([100], "lastBiases")
        # lastWeights = init_weights((100, 100), "lastWeigts")
        # lastLayer = maxout.max_out(tf.nn.relu(tf.add(tf.matmul())))



    init = tf.global_variables_initializer()
    sess.run(init)

    index = 0

    y = tf.placeholder("float", shape=[None, 1], name="y")

    # dict = populate_dict(0, y)
    # print("output is: " + str(sess.run(outLayer, feed_dict=dict)))
    cost = tf.losses.mean_squared_error(y, outLayer)
    scopeVars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="local")
    print("printing vars")
    for v in scopeVars:
        print("v in vars is: " + str(v))

    print("printing vars2")
    currentLearnRate = .1
    learn_rate = tf.placeholder("float")
    # updates = tf.train.AdamOptimizer(name="optimizer").minimize(cost)
    updates1 = tf.train.GradientDescentOptimizer(learn_rate,name="optimizer").minimize(cost)
    heroIndex = 0
    for hero in heroes:
        print("top of loop: " + hero)
        scopeVars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=hero)
        for v in scopeVars:
            print("hero vars:" + str(v))

        updates1 = tf.group(updates1, tf.train.GradientDescentOptimizer(
            learn_rate,
            # tf.multiply(tf.multiply(learn_rate, scaleDownHeroRate), tf.add(.05,tf.reduce_mean(tf.gather(tf.transpose(percentTimePlayed), [heroIndex])))),
            name="optimizer1" + str(hero)).minimize(cost, var_list=scopeVars))
        heroIndex = heroIndex + 1

    cost2 = tf.losses.absolute_difference(y, outLayer)
    updates2 = tf.train.GradientDescentOptimizer(learn_rate, name="optimizer").minimize(cost2)
    heroIndex = 0
    for hero in heroes:
        print("top of loop: " + hero)
        scopeVars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=hero)
        for v in scopeVars:
            print("hero vars:" + str(v))

        updates2 = tf.group(updates2, tf.train.GradientDescentOptimizer(
            learn_rate,

            # tf.multiply(tf.multiply(learn_rate, scaleDownHeroRate),tf.add(tf.reduce_mean(tf.gather(tf.transpose(percentTimePlayed), [heroIndex])),.05)),
            name="optimizer1" + str(hero)).minimize(cost2, var_list=scopeVars))
        heroIndex = heroIndex + 1
    #    updates = tf.group(updates,tf.Print(tf.Variable(0),"hello"))
    average = 540
    saver = tf.train.Saver()
    if saverDir != None:
        print("hello")
        # saver.restore(sess,"/home/dfreelan/data/data/tensorModels/" + "fastLRMaxout2Out_oldRelu_FORCED3allPerGameMedalsQUADNEURONSNoNegativeWithBias2MoreIncludeTimeFineTuned-11400")# saverDir+"_activeSeason")
        # saver.restore(sess,"/home/dfreelan/data/data/tensorModels/" + "fastLRMaxout2Out_oldRelulatestDataMoreAccurate-12100")# saverDir+"_activeSeason")

    else:
        saverIndex = 0
    saver = tf.train.Saver()
    for i in range(20000000):
        i = i+saverIndex
        currentLearnRate = 2.0/(0.025*i+7.752179328)
        # currentLearnRate=.1
        # print(currentLearnRate)
        # if i >= 10:
        #     currentLearnRate = .08
        # if i >= 50:
        #     currentLearnRate = .03
        # if i >= 100:
        #     currentLearnRate = .015
        # if i >= 220:
        #     currentLearnRate = .0075
        # if i >= 500:
        #     currentLearnRate = .005


        updates = updates1
        currentLearnRate = 3.5 / (0.025 * i + 7.752179328)

        i = i-saverIndex
        for k in range(100):
            allInputBatch, labelsBatch, weightsBatch = create_all_hero_batch(128, True, labels)
            feed_dict = convertToDictionary(allInputBatch, labelsBatch, weightsBatch, currentLearnRate,
                                            percentTimePlayed, y, learn_rate, keep_prob, .5)
            sess.run(updates, feed_dict=feed_dict)

        allInputBatch, labelsBatch, weightsBatch = create_all_hero_batch(2000, False,labels)
        feed_dict = convertToDictionary(allInputBatch, labelsBatch, weightsBatch, currentLearnRate,
                                        percentTimePlayed, y, learn_rate, keep_prob, 1.0)
        test_accuracy = np.average(
            np.abs(labelsBatch - sess.run(outLayer, feed_dict=feed_dict))
        )
        allInputBatch, labelsBatch, weightsBatch = create_all_hero_batch(2000, True,labels)
        feed_dict = convertToDictionary(allInputBatch, labelsBatch, weightsBatch, currentLearnRate,
                                        percentTimePlayed, y, learn_rate, keep_prob, 1.0)
        train_accuracy = np.average(
            np.abs(labelsBatch - sess.run(outLayer, feed_dict=feed_dict))
        )
        # print("percent played: " + str(sess.run(percentTimePlayed,feed_dict=feed_dict)))
        # print("average ana?: " + str(sess.run(tf.reduce_mean(tf.gather(tf.transpose(percentTimePlayed),[0])),feed_dict=feed_dict)))
        i = i + saverIndex
        print("test " + fileName + ": " + str(test_accuracy * 5000.0) + " " + str(i) + " " + str(train_accuracy * 5000.0))
        # np.set_printoptions(threshold=np.inf)
       # print(labelsBatch - sess.run(outLayer, feed_dict=feed_dict))
        if i%100 == 0:
            # print("saving..." + str(i))
            # print("saving..." + str(i))
            saver.save(sess, datadir + "tensorModels/" + fileName+"season10Model", global_step=i)
        i = i - saverIndex
        # dict = populate_dict(np.random.randint(0, len(labels[0]) * .9), y)
        # sess.run(updates, feed_dict=dict)
        # dict = populate_dict(np.random.randint(len(labels[0]) * .9, len(labels[0])), y)
        # srError = 5000.0 * sess.run(cost, feed_dict=dict)
        # average = .001 * srError + average * .999
        # if i % 1000 == 0:
        #     print("error: " + str(srError) + "avg: " + str(average) + " label: " + str(5000.0*sess.run(y,feed_dict=dict)[0][0]))

        # cost = tf.multiply(tf.losses.absolute_difference(y, yhat), weight,name="cost")
        # updates = tf.train.GradientDescentOptimizer(learn_rate,name="optimizer").minimize(cost)
        # print("did i find prediction? " + str(tf.get_local_variable("prediction")))
        # tf.concat([outputVec,modelVars])

        # all_vars = tf.all_variables()
        # model_one_vars = [k for k in all_vars if k.name.startswith("en_fr")]
        # model_two_vars = [k for k in all_vars if k.name.startswith("fr_en")]

        # if not model_two_vars:
        #   self.saver_en_fr = tf.train.Saver(model_one_vars)
        # else:
        #   self.saver_fr_en = tf.train.Saver(model_two_vars)

def create_bias_variable(name, shape):
    initializer = tf.constant_initializer(value=0.0, dtype=tf.float32)
    return tf.Variable(initializer(shape=shape), name=name)

def create_bias_variable(shape):
    initializer = tf.constant_initializer(value=0.0, dtype=tf.float32)
    return tf.Variable(initializer(shape=shape))

def concat_networks(outputVec, secondOutput):
    global sess
    for heroIndex in range(len(heroes)):
        saver = None
        if withTrain:
            saver = tf.train.import_meta_graph("/home/dfreelan/data/data/tensorModels/" + heroes[heroIndex] + "season9.5-2000.meta")
        else:
            saver = tf.train.import_meta_graph(
                "/home/dfreelan/data/data/tensorModelsNoTrain/" + heroes[heroIndex] + "-2000.meta")
        saver.restore(sess, datadir + "tensorModels/" + heroes[heroIndex] + "season9.5-2000")
        # print(sess.run("ana/prediction:0"))
        graph = tf.get_default_graph()
        predictor = graph.get_tensor_by_name(heroes[heroIndex] + "/prediction:0")
        X = graph.get_tensor_by_name(heroes[heroIndex] + "/X:0")
        keep_prob = graph.get_tensor_by_name(heroes[heroIndex] + "/keep_prob:0")
        if secondOutput:
            with tf.variable_scope(heroes[heroIndex]):
                w1 = graph.get_tensor_by_name(heroes[heroIndex] + "/" + heroes[heroIndex] + "w1:0")
                w2 = graph.get_tensor_by_name(heroes[heroIndex] + "/" + heroes[heroIndex] + "w2:0")
                lastOutput = graph.get_tensor_by_name(heroes[heroIndex] + "/" + heroes[heroIndex] + "w3:0")
                secondOutput= init_weights([50,1],wName="out2")

                copy_first_variable = secondOutput.assign(lastOutput)
                init = tf.initialize_all_variables()

                sess.run(init)
                newOutput=tf.nn.relu(tf.matmul(w2,secondOutput))
                init_new_vars_op = tf.initialize_variables([secondOutput])
                sess.run(init_new_vars_op)
                predictor = forwardprop(X,w1,w2,lastOutput,secondOutput)
        # if heroes[heroIndex]=='ana':
        #      print("shape is: " + str(sess.run(tf.shape(predictor),feed_dict={X:np.full(fill_value=.999,shape=[3,18])})))
        #      print("shape2 is: " + str(
        #          sess.run(tf.shape(newOutput), feed_dict={X: np.full(fill_value=.999, shape=[3, 18])})))
        #         # init_new_vars_op = tf.initialize_variables([lastOutput])
        #         # sess.run(init_new_vars_op)


        keepProbs.append(keep_prob)
        heroInput.append(X)
        if outputVec != None:
            outputVec = tf.concat([outputVec, predictor], 1)
            print("shape of output: " + str(tf.shape(outputVec)))
        else:
            outputVec = (predictor)
            print("shape of output2: " + str(tf.shape(outputVec)))
    return outputVec


def concat_networkss(outputVec, secondOutput,percentVec):
    global sess
    count = 0
    percentVec = tf.transpose(percentVec)
    for heroIndex in range(len(heroes)):
        if withTrain:
            saver = tf.train.import_meta_graph("/home/dfreelan/data/data/tensorModels/" + heroes[heroIndex] + "3OutFORCEDperGameMedals-2000.meta")
        else:
            saver = tf.train.import_meta_graph(
                "/home/dfreelan/data/data/tensorModelsNoTrain/" + heroes[heroIndex] + "-2000.meta")
        saver.restore(sess, datadir + "tensorModels/" + heroes[heroIndex] + "3OutFORCEDperGameMedalsWithBias-2000")
        # print(sess.run("ana/prediction:0"))
        graph = tf.get_default_graph()
        predictor = graph.get_tensor_by_name(heroes[heroIndex] + "/prediction:0")
        X = graph.get_tensor_by_name(heroes[heroIndex] + "/X:0")
        keep_prob = graph.get_tensor_by_name(heroes[heroIndex] + "/keep_prob:0")
        if secondOutput:
            with tf.variable_scope(heroes[heroIndex]):
                w1 = graph.get_tensor_by_name(heroes[heroIndex] + "/" + heroes[heroIndex] + "w1:0")
                w2 = graph.get_tensor_by_name(heroes[heroIndex] + "/" + heroes[heroIndex] + "w2:0")
                lastOutput = graph.get_tensor_by_name(heroes[heroIndex] + "/" + heroes[heroIndex] + "w3:0")
                b1 = graph.get_tensor_by_name(heroes[heroIndex] + "/" + heroes[heroIndex] + "b1:0")
                b2 = graph.get_tensor_by_name(heroes[heroIndex] + "/" + heroes[heroIndex] + "b2:0")
                b3 = graph.get_tensor_by_name(heroes[heroIndex] + "/" + heroes[heroIndex] + "b3:0")
                secondOutput= init_weights([50,1],wName="out2")

                copy_first_variable = secondOutput.assign(lastOutput)
                init = tf.initialize_all_variables()

                sess.run(init)
                newOutput=tf.nn.relu(tf.matmul(w2,secondOutput))
                init_new_vars_op = tf.initialize_variables([secondOutput])
                sess.run(init_new_vars_op)
                predictor = forwardprop(X,w1,w2,lastOutput,secondOutput,keep_prob)
        # if heroes[heroIndex]=='ana':
        #      print("shape is: " + str(sess.run(tf.shape(predictor),feed_dict={X:np.full(fill_value=.999,shape=[3,18])})))
        #      print("shape2 is: " + str(
        #          sess.run(tf.shape(newOutput), feed_dict={X: np.full(fill_value=.999, shape=[3, 18])})))
        #         # init_new_vars_op = tf.initialize_variables([lastOutput])
        #         # sess.run(init_new_vars_op)


        keepProbs.append(keep_prob)
        heroInput.append(X)
        predictor=tf.transpose(predictor)
        if outputVec != None:

            for i in range(3):
                weight = create_bias_variable([1])
                outputVec = tf.concat([outputVec, tf.multiply(tf.gather_nd(predictor, [i]),
                                                             tf.add(tf.gather_nd(percentVec, [count]), weight))],1)

            print("shape of output: " + str(tf.shape(outputVec)))
        else:
            weight = create_bias_variable([1])
            outputVec=tf.get_variable(name="heroconcatcrap",shape=[128,0])
            outputVec = tf.concat([outputVec,tf.multiply(tf.gather_nd(predictor, [0]),tf.add(tf.gather_nd(percentVec,[count]),weight))],1)
            for i in range(2):
                weight = create_bias_variable([1])
                outputVec = tf.concat([outputVec,tf.multiply(tf.gather_nd(predictor, [i+1]),tf.add(tf.gather_nd(percentVec,[count]),weight))],1)
                print("shape of output2: " + str(tf.shape(outputVec)))
        count = count+1
    return outputVec



# def forwardprop(X, w_1, w_2, w_3, w_32,keep_prop=1.0, batch_size=128):
#     """
#     Forward-propagation.
#     IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
#     """
#     resultMultiplier = tf.floor(tf.add(tf.reduce_mean(X,axis=1),1.0))
#     h1 = tf.nn.relu(tf.matmul(X, w_1))  # The \sigma function
#     h2 = tf.nn.relu(tf.matmul(h1, w_2))
# 
#     yhat = tf.matmul(h2, w_3)  # The \varphi function
#     yhat2 = tf.nn.sigmoid(yhat, name="prediction")
# 
#     additionOutput = tf.nn.sigmoid(tf.matmul(h2, w_32))
#     # averages = tf.reduce_mean(X, axis=0)
#     # results = tf.zeros(shape=[batch_size, 1])
#     # 
#     # # scatter = None
#     # 
#     # for k in range(batch_size):
#     #     # tf.gather(results,[k]) = tf.cond(tf.gather(averages,[k]))
#     #     indices = tf.constant(k, shape=[1, 1],dtype="int32")
#     #     updateValue =  tf.cond(tf.less(tf.gather(averages, tf.constant(k, dtype="int32")),tf.constant(0.0)),
#     #                                 lambda: tf.constant(1),# tf.gather(tf.gather(averages, tf.constant(k, shape=[1, 1],dtype="int32")),[0,0]),
#     #                                 lambda: tf.constant(0))# tf.gather(tf.gather(yhat2, tf.constant([tf.constant(k,dtype="int32"),tf.constant(0,dtype="int32")],dtype="int32"))),[0,0])
#     # 
#     #     updates = updateValue
#     #     shape = tf.shape(results)
#     #     if scatter == None:
#     #         scatter = tf.scatter_nd(indices, updates, shape)
#     #     else:
#     #         scatter = tf.group([scatter,tf.scatter_nd(indices, updates, shape)],1)
#     # print("building shape is " + str(tf.Session().run(yhat2, feed_dict={X:np.zeros([2,18])})))
#     return tf.concat([tf.transpose(tf.multiply(tf.transpose(yhat2),resultMultiplier)),tf.transpose(tf.multiply(tf.transpose(additionOutput),resultMultiplier))],1)
def forwardprop(X, w_1, w_2, w_3,w_32, keep_prop, batch_size=128):
    """
    Forward-propagation.
    IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
    """
    # resultMultiplier = tf.floor(tf.add(tf.reduce_mean(X,axis=1),1.0))
    # X=tf.layers.batch_normalization(X,training=tf.less(keep_prop,tf.constant(1.0)))
    h1 = tf.nn.relu(tf.matmul(X, w_1))  # The \sigma function
    # h1 = maxout.max_out(h1,200)
    # h1 = tf.layers.batch_normalization(h1,training=(keep_prop!=1.0))
    # h1 = tf.nn.batch_normalization(h1,0,2,0.0,1.0,.0001,)
    # .layer(1,new BatchNormalization.Builder().nIn(numHidden).nOut(numHidden).gamma(2).beta(0.001).decay(.99).updater(Updater.SGD).build())

   # h1 = tf.layers.batch_normalization(h1, training=tf.less(keep_prop,tf.constant(1.0)))
    h2 = tf.nn.relu(tf.matmul(h1, w_2))
    # print("blah :" + str( keep_prop))
    h2 = tf.nn.dropout(h2,keep_prop)
    # h2 = maxout.max_out(h2, 50)
    # h2 = maxout.max_out(h2, 200)
    # h2 = tf.nn.dropout(h2, keep_prob=keep_prop)

    yhat = tf.matmul(h2, w_3)  # The \varphi function
    # yhat=tf.layers.batch_normalization(yhat, training=(keep_prop != 1.0))
    yhat2 = tf.nn.sigmoid(yhat, name="prediction")

    additionOutput = tf.nn.sigmoid(tf.matmul(h2, w_32))
    # averages = tf.reduce_mean(X, axis=0)
    # results = tf.zeros(shape=[batch_size, 1])
    # 
    # # scatter = None
    # 
    # for k in range(batch_size):
    #     # tf.gather(results,[k]) = tf.cond(tf.gather(averages,[k]))
    #     indices = tf.constant(k, shape=[1, 1],dtype="int32")
    #     updateValue =  tf.cond(tf.less(tf.gather(averages, tf.constant(k, dtype="int32")),tf.constant(0.0)),
    #                                 lambda: tf.constant(1),# tf.gather(tf.gather(averages, tf.constant(k, shape=[1, 1],dtype="int32")),[0,0]),
    #                                 lambda: tf.constant(0))# tf.gather(tf.gather(yhat2, tf.constant([tf.constant(k,dtype="int32"),tf.constant(0,dtype="int32")],dtype="int32"))),[0,0])
    # 
    #     updates = updateValue
    #     shape = tf.shape(results)
    #     if scatter == None:
    #         scatter = tf.scatter_nd(indices, updates, shape)
    #     else:
    #         scatter = tf.group([scatter,tf.scatter_nd(indices, updates, shape)],1)
    # print("building shape is " + str(tf.Session().run(yhat2, feed_dict={X:np.zeros([2,18])})))
    return tf.concat([yhat2,
                      additionOutput ])

main()



