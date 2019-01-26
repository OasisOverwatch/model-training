import argparse

import tensorflow as tf
import numpy as np
from os import listdir
from os.path import isfile, join
import threading
import Maxout as maxout
import socketserver
import socket

datadir = "/data"
listofFiles = listdir(datadir)
listofFiles.sort()
onlyfiles = [f for f in listofFiles if isfile(join(datadir, f))]
inputData = []
count = 0
heroCount = 0

heroes = []
for file in onlyfiles:
    if (".dat" in file and not ".data" in file):
        with open(datadir + file, 'r') as f:
            if (not ("skill.dat" in file) and not "skillNormal.dat" in file):
                print(file)
                content = f.readlines()
                inputData.append(np.zeros(shape=[len(content), len(content[0].split())]))
                heroCount = heroCount + 1
                heroes.append(file[0:-4])
                count = 0
                for x in content:
                    if (len(x.split()) != 0):
                        inputData[heroCount - 1][count] = (np.asarray([(float(x)) for x in x.split()]))
                        count = count + 1
                if (heroCount > 100):
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
    if (".weight" in file):
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
            if (heroCount > 100):
                break
labelsPercentiled = []


# RANDOM_SEED = 42
# tf.set_random_seed(RANDOM_SEED)
sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=64))


def init_weights(shape, wName):
    """ Weight initialization """
    weights = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(weights, name=wName)


doMaxOut = True
withTrain = True
modelVars = []
percentTimePlayed  = ""
heroInput = []
keepProbs = []
heroLearnAdjustment = {}
scaleDownHeroRate = 1.0
# tf.placeholder("float",
for hero in heroes:
    heroLearnAdjustment[hero] = (tf.placeholder("float", shape=[1]))
trainingEndIndex = len(inputData[0]) -1
print("training set index is: " + str(trainingEndIndex))

# shape should be....
# [hero][batch_size][inputsize]

def create_all_hero_batch(batch_size, isTrain, labels):
    # heroInput = np.zeros(batch_size)
    # keepProbs = np.zeros(batch_size)
    allHeroBatchData = []
    allHeroBatchLabels = []
    allHeroBatchWeights = np.zeros(shape=[batch_size, len(heroes)])
    if (not isTrain):
        allHeroBatchWeights = np.zeros(shape=[len(inputData[0])-int(np.floor(trainingEndIndex)), len(heroes)])
    for hero in range(len(heroes)):
        allHeroBatchData.append([])
        if(isTrain):
            allHeroBatchData[hero] = np.zeros(shape=[batch_size, len(inputData[hero][0])])
        else:
            allHeroBatchData[hero] = np.zeros(shape=[len(inputData[0])-int(np.floor(trainingEndIndex)), len(inputData[hero][0])])
    if(not isTrain):
        for i in range(len(inputData[0])-int(np.floor(trainingEndIndex))):
            randIndex = int(np.floor(trainingEndIndex))+i
            allHeroBatchLabels.append(labels[0][randIndex])

            for hero in range(len(heroes)):
                allHeroBatchData[hero][i] = inputData[hero][randIndex]
                allHeroBatchWeights[i][hero] = weightsRaw[hero][randIndex]

        return allHeroBatchData, allHeroBatchLabels, allHeroBatchWeights
    for i in range(batch_size):
        randIndex = 0

        if (isTrain):
            randIndex = np.random.randint(0, trainingEndIndex)
        else:
            randIndex = np.random.randint(trainingEndIndex, len(inputData[0]))

        allHeroBatchLabels.append(labels[0][randIndex])

        for hero in range(len(heroes)):
            allHeroBatchData[hero][i] = inputData[hero][randIndex]
            allHeroBatchWeights[i][hero] = weightsRaw[hero][randIndex]

    return allHeroBatchData, allHeroBatchLabels, allHeroBatchWeights


def convertToDictionary(batchData, batchWeights):
    myDict = {}
    for i in range(len(heroInput)):
        # print("i is: " + str(i) + " " + str(len(batchData)) + " len heroInput " + str(len(heroInput)))
        myDict[heroInput[i]] = np.asarray([batchData[i]])
        myDict[keepProbs[i]] = 1.0
    myDict[percentTimePlayed] = np.asarray([batchWeights])
    return myDict
def convertToDictionaryPrebatched(batchData, batchWeights):
    myDict = {}
   # print("len of hero input"  + str(len(heroInput)) + " len batch data " + str(len(batchData)))
    for i in range(len(heroInput)):
        myDict[heroInput[i]] = batchData[i]
        myDict[keepProbs[i]] = 1.0
    myDict[percentTimePlayed] = batchWeights
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
outLayer = ""
def main():
    # saver = tf.train.Saver()
    global secondOutput
    global doMaxOut
    global scaleDownHeroRate
    global withTrain
    global outLayer
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
    if((not secondOutput) and (not doMaxOut) and scaleDownHeroRate==1.0 and withTrain):
        fileName = "concatedControlOldLearnNTNormArchitectWITHTRAINFAST"
    if(secondOutput and not doMaxOut and scaleDownHeroRate==1.0 and withTrain):
        fileName = "concatedControlOldLearnNTNormArchitectWITHTRAINFASTSEecondOutput"
    if (not secondOutput and not doMaxOut and scaleDownHeroRate == .1 and withTrain):
        fileName = "concatedControlOldLearnNTNormArchitectWITHTRAIN"
    # concatedControlOldLearnNTExpArchitectWITHTRAIN: 266.394986335
    # 36936
    # 249.63537267
    # test fastLRMaxout: 267.711412895 1200 264.994412072
    # test fastLRMaxout: 246.992130535 2800 241.610342251
    # test fastLRMaxout2Out: 242.229462524 11500 225.887557329
    # test fastLRMaxout: 244.137286386 9086 233.993045182
    if (not secondOutput and doMaxOut and scaleDownHeroRate == .1 and withTrain):
        fileName = "concatedControlOldLearnNTExpArchitectWITHTRAIN"
    if(not secondOutput and doMaxOut and scaleDownHeroRate == 1.0 and withTrain):
        fileName = "fastLRMaxout2Out"
    if (secondOutput and doMaxOut and scaleDownHeroRate == .1 and withTrain):
        fileName = "slowSecondOutMaxOut"
    if (secondOutput and doMaxOut and scaleDownHeroRate == 1.0 and withTrain):
        fileName = "fastSecondOutMaxOut"
    # withTrain = True
    print("with train is " + str(withTrain))
    print("file name is " + fileName)
    if(fileName==None):
        return
    # fileName = "concatedControlOldLearnNTNormArchitectWITHTRAINFASTSEecondOutput"
    saverDir = datadir + "tensorModels/" + fileName + "-" + str(saverIndex)
    if(saverIndex==0):
        saverDir = None
    outputVec = None
    outputVec = concat_networks(outputVec,secondOutput)

    percentTimePlayed = tf.placeholder("float", shape=[None, len(heroes)], name="percentTimePlayed")
    keep_prob = tf.placeholder("float")
    # print("shape before: " +sess.run(tf.shape(outputVec)))
    outputPlusPercent = tf.concat([percentTimePlayed, outputVec], 1)
    with tf.variable_scope("local"):
        multiplier = 2
        multiplier = 4
        concatWeight = init_weights((27 * multiplier, 400), "concatWeight")
        print("concat weight is initialized right? " + str(concatWeight))
        concatedLayerBiases = init_weights([400], "concatBias")
        concatedLayer = maxout.max_out(
            tf.nn.relu(tf.add(tf.matmul(outputPlusPercent, concatWeight), concatedLayerBiases)), 200)

        midWeights = init_weights((200, 200), "midWeight")
        if (doMaxOut):
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


    init = tf.global_variables_initializer()
    sess.run(init)

    index = 0

    y = tf.placeholder("float", shape=[None, 1], name="y")

    # dict = populate_dict(0, y)
    # print("output is: " + str(sess.run(outLayer, feed_dict=dict)))
    cost = tf.losses.absolute_difference(y, outLayer)
    scopeVars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="local")

    #    updates = tf.group(updates,tf.Print(tf.Variable(0),"hello"))
    average = 540
    saver = tf.train.Saver()
    if(saverDir != None):
       # 
    #  saver.restore(sess, "/home/dfreelan/data/data/tensorModels/" + "fastLRMaxout2Out_oldRelu_FORCED3allPerGameMedalsQUADNEURONSNoNegativeWithBias2MoreIncludeTimeFineTuned-11400")
     saver.restore(sess, "/home/dfreelan/data/data/tensorModels/fastLRMaxout2Out_oldReluseason10Model-1300")
    else:
        saverIndex = 0
def appendCloneToHeroData(heroData, clone):
    # print("size of hero data is " + str(len(heroData)))
    for i in range(len(clone)):
        heroData[i].append(clone[i])
# [[7, 12], [7, 5], [7, 22], [7, 23], [7, 4], [7, 6], [7, 18], [7, 20], [7, 16], [7, 10], [7, 2], [7, 1], [7, 13], [7, 15], [7, 14], [7, 11], [7, 17], [7, 3], [7, 0], [7, 7], [7, 24], [7, 9], [7, 8], [7, 8], [7, 9], [7, 19], [7, 21], [7, 22], [7, 24], [7, 7], [7, 0], [7, 3], [7, 17], [7, 11], [7, 14], [7, 19], [7, 15], [7, 13], [7, 1], [7, 2], [7, 10], [7, 21], [7, 16], [7, 20], [7, 18], [7, 6], [7, 4], [7, 23], [7, 5], [7, 12]]


def ZeroWeightExcept(weights, param, heroData):
    newWeights = []
    newHeroData=[]
    for i in range(len(weights)):
        newHeroData.append(heroData[i])
        if(i!=param):
            # if(weights[i]<.1):
               # newWeights.append(weights[i])
            # else:
             #   print("this never happens")
                newWeights.append(weights[i]/2.0)
        else:
            w = weights[i]*2
            if(w>1.0):
                w = 1.0
            if(w<.5):
                w=.5
            newWeights.append(w)
            w = weights[i] * 2


            w = newHeroData[i][len(heroData[i]) - 1]
            if (w > 1.0):
                w = 1.0
            if (w < .5):
                w = .5
            newHeroData[i][len(heroData[i]) - 1] = w

            w = newHeroData[i][len(heroData[i]) - 2]
            if (w > 1.0):
                w = 1.0
            if (w < .5):
                w = .5
            newHeroData[i][len(heroData[i]) - 2] = w

    return newWeights,newHeroData

def createHeroBatchRequest(heroData,weights,requests):
    newHeroData = []
    newWeights = []
    modificationLocations = []
    modificationCount = 0

    for i in range(len(heroData)):
        newHeroData.append([])
        for a in range(len(heroData[i])):
            if (heroData[i][a] != -1):
                if (i == requests[0]):
                    modificationCount += 1
    for i in range(len(heroData)):
        if (heroData[i][0] != -1):
            newHeroData.append(np.zeros(shape=(modificationCount, len(heroData[i]))))
            for a in range(len(heroData[i])):
                if (i == requests[0]):
                    modificationLocations.append([i, a])

    for mod in modificationLocations:
        clone = cloneExceptIndex(heroData, mod[0], mod[1], .01)
        appendCloneToHeroData(newHeroData, clone)
        newWeights.append(weights)
       # clone = cloneExceptIndex(heroData, mod[0], mod[1], -.01)
       # appendCloneToHeroData(newHeroData, clone)
       # newWeights.append(weights)
    return modificationLocations, newHeroData, newWeights


def createHeroBatch(heroData, weights):
    # count how many entries we're going to have to each hero (will be the same...)
    # make each index of newHeroData an np array with that size
    # dimemsions should be newHeroData[hero][modificationIteration][feature]
    newHeroData = []
    newWeights = []
    modificationLocations = []
    modificationCount = 0

    for i in range(len(heroData)):
        newHeroData.append([])
        for a in range(len(heroData[i])):
            if(heroData[i][a] !=-1):
                modificationCount+=1
    for i in range(len(heroData)):
        if(heroData[i][0]!=-1):
            newHeroData.append(np.zeros(shape=(modificationCount,len(heroData[i]))))
            for a in range(len(heroData[i])):
                modificationLocations.append([i,a])

    for mod in modificationLocations:
        clone = cloneExceptIndex(heroData,mod[0],mod[1],.1)
        appendCloneToHeroData(newHeroData,clone)
        newWeights.append(weights)
        clone = cloneExceptIndex(heroData, mod[0], mod[1], -.1)
        appendCloneToHeroData(newHeroData, clone)
        newWeights.append(weights)
    return modificationLocations,newHeroData,newWeights


def createHeroBatchOneTrick(heroData, weights):
    # count how many entries we're going to have to each hero (will be the same...)
    # make each index of newHeroData an np array with that size
    # dimemsions should be newHeroData[hero][modificationIteration][feature]
    newHeroData = []
    newWeights = []
    modificationLocations = []
    modificationCount = 0

    for i in range(len(heroData)):
        newHeroData.append([])
        for a in range(len(heroData[i])):
            if (heroData[i][a] != -1):
                modificationCount += 1
    for i in range(len(heroData)):
        if (heroData[i][0] != -1):
            newHeroData.append(np.zeros(shape=(modificationCount, len(heroData[i]))))
            for a in range(len(heroData[i])):
                modificationLocations.append([i, a])

    for mod in modificationLocations:
        clone = cloneExceptIndexOnTime(heroData, mod[0], mod[1], .0)
        appendCloneToHeroData(newHeroData, clone)
        newWeights.append(ZeroWeightExcept(weights,mod[0]))
        clone = cloneExceptIndexOnTime(heroData, mod[0], mod[1], -.0)
        appendCloneToHeroData(newHeroData, clone)
        newWeights.append(ZeroWeightExcept(weights,mod[0]))
        if(mod[0] == 6):
            print("the new weights? " + str(ZeroWeightExcept(weights,mod[0])))
    return modificationLocations, newHeroData, newWeights
def createHeroBatchNoModification(heroData,weights,request):
    newHeroData = []
    newWeights = []
    for i in range(len(heroData)):
        newHeroData.append([])
    w,modifiedHeroData= ZeroWeightExcept(weights, request, heroData)

    clone = cloneExceptIndexOnTime(modifiedHeroData, 0, 0, .0)

    appendCloneToHeroData(newHeroData, clone)

    newWeights.append(w)


    return  newHeroData, newWeights
def createHeroBatchOneTrickRequested(heroData, weights,requests):
    # count how many entries we're going to have to each hero (will be the same...)
    # make each index of newHeroData an np array with that size
    # dimemsions should be newHeroData[hero][modificationIteration][feature]
    newHeroData = []
    newWeights = []
    modificationLocations = []
    modificationCount = 0

    for i in range(len(heroData)):
        newHeroData.append([])
        for a in range(len(heroData[i])):
            if (heroData[i][a] != -1):
                if (i == requests[0]):
                    modificationCount += 1
    for i in range(len(heroData)):
        if (heroData[i][0] != -1):
            newHeroData.append(np.zeros(shape=(modificationCount, len(heroData[i]))))
            for a in range(len(heroData[i])):
                if(i ==requests[0]):
                    modificationLocations.append([i, a])

    for mod in modificationLocations:
        print(" mod is  " + str(mod))
        clone = cloneExceptIndexOnTime(heroData, mod[0], mod[1], .01)
        appendCloneToHeroData(newHeroData, clone)
        newWeights.append(ZeroWeightExcept(weights, mod[0]))
        # clone = cloneExceptIndexOnTime(heroData, mod[0], mod[1], -.01)
        # appendCloneToHeroData(newHeroData, clone)
        newWeights.append(ZeroWeightExcept(weights, mod[0]))
        if (mod[0] == 6):
            print("the new weights? " + str(ZeroWeightExcept(weights, mod[0])))
    return modificationLocations, newHeroData, newWeights


def cloneExceptIndex(data,x,y,epsilon):
    clone = []
    for i in range(len(data)):
        clone.append(np.copy(data[i]))
        if(i==x):
            # from scipy.stats import norm
            # inversed = norm.cdf(norm.ppf(clone[x][y])*epsilon)
            clone[x][y] +=epsilon
            if(clone[x][y]<0):
                clone[x][y] = 0
    return clone
def cloneExceptIndexOnTime(data,x,y,epsilon):
    clone = []
    for i in range(len(data)):

        # if(i!=x):
        clone.append(np.copy(data[i]))
        if(i==x):
            # clone.append(np.copy(data[i]))
            clone[x][y] +=epsilon
            if(clone[x][y]<0):
                clone[x][y] = 0
        # if(i==6):
        #    print("clone at six is " + str(clone[6]))
    return clone
class struct_instance:


    def __init__(self):
        self.topN = 600
        self.topListSr = []
        self.topListLocation = []
        self.topListDirections = []
        self.baseline = 0.0
class ThreadedServer(object):
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))

    def listen(self):
        self.sock.listen(5)
        while True:
            client, address = self.sock.accept()
            client.settimeout(60)

            threading.Thread(target = self.listenToClient,args = (client,address)).start()

    def insert(self, localData, newSr, location, dir):
        insertIndex = -1
        for i in range(len(localData.topListSr)):
            if (localData.topListSr[i] < newSr):
                insertIndex = i
                break
        if (insertIndex != -1):
            localData.topListSr.insert(insertIndex, newSr)
            localData.topListLocation.insert(insertIndex, location)
            localData.topListDirections.insert(insertIndex, dir)
        else:
            localData.topListSr.append(newSr)
            localData.topListLocation.append(location)
            localData.topListDirections.append(dir)

    def maintainList(self, localData, newSr, location, dir,weight):
        # print("dir is : " +
        # print("new sr is " + str(newSr))
        newSr = 50000.0*(newSr-localData.baseline)# *weight
        if (len(localData.topListSr) < localData.topN):
            self.insert(localData, newSr, location, dir)
        else:
            if (localData.topListSr[localData.topN - 1] < newSr):
                self.insert(localData, newSr, location, dir)
                del localData.topListLocation[localData.topN]
                del localData.topListSr[localData.topN]
                del localData.topListDirections[localData.topN]

    def readline(self,request):
        char = request.recv(1).decode("utf-8")
        builder = ""
        while (char != "\n"):
            builder += str(char)
            # print(builder)
            char = request.recv(1).decode("utf-8")
        return builder

    def listenToClient(self, client, address):
        global outLayer
        # self.request is the TCP socket connected to the client

        print("waitig on first line")
        data = self.readline(client)
        print("got first line: " + str(data))
        heroData = []
        heroLines = data.split("")
        messageBuilder = ""
        # get player features
        for heroLine in heroLines:
            heroData.append((np.asarray([(float(x)) for x in heroLine.split(",")])))
        # get weights
        data = self.readline(client)
        print("got second line: " + str(data))
        weights = np.asarray([(float(x)) for x in data.split(",")])

        # get requests
        data = self.readline(client)
        print("got third line: " + str(data))
        requests = np.asarray([(float(x)) for x in data.split(",")])

        # oasis sr
        baseline = sess.run(outLayer, feed_dict=convertToDictionary(heroData, weights))[0][0]
        print("baseline is: " + str(baseline))
        messageBuilder += str(baseline) + "\n"
        for req in requests:
            print("on hero " + str(req))
            localData = struct_instance()
            # baseline for hero
            heroDataLocal, weightsLocal = createHeroBatchNoModification(heroData, weights,req)
            heroBaseline=sess.run(outLayer, feed_dict=convertToDictionaryPrebatched(heroDataLocal, weightsLocal))

            localData.baseline=baseline
            # localData.baseline=heroBaseline[0][0]
            messageBuilder += str(heroBaseline[0][0]) + "\n"
            print("SR1 for hero is " + str((localData.baseline)))
            print("SR2 for hero is " + str((heroBaseline[0][0])))
            featureIdentifiers, heroDataLocal, weightsLocal = createHeroBatchRequest(heroData, weights, [req])
            allResults = sess.run(outLayer, feed_dict=convertToDictionaryPrebatched(heroDataLocal, weightsLocal))
            print("SR3 for hero is " + str((localData.baseline)))
            print("SR4 for hero is " + str((heroBaseline[0][0])))
            for i in range(len(allResults)):
                location = featureIdentifiers[i]

                self.maintainList(localData, allResults[i][0], featureIdentifiers[i], 1,weight=1)# ,np.power(weights[i][featureIdentifiers[i//2][0]],1))

            print("list of indexeS:" + str(localData.topListLocation))
            print("improvements: " + str(localData.topListSr))
            print("directions: " + str(localData.topListDirections))
            # just send back the same data, but upper-cased
            messageBuilder+=str(localData.topListLocation) + "\n" + str(localData.topListSr) + "\n"  + str(localData.topListDirections) + "\n"
        print("sending: " + (messageBuilder))
        client.sendall(str.encode(messageBuilder))

class MyTCPHandler(socketserver.BaseRequestHandler):
    """
    The RequestHandler class for our server.

    It is instantiated once per connection to the server, and must
    override the handle() method to implement communication to the
    client.
    """



    def handle(self):
      t = threading.Thread(target=self.handleRequest)
      t.start()





def concat_networks(outputVec, secondOutput):
    global sess
    for heroIndex in range(len(heroes)):
        saver = None
        if(withTrain):
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
        if(secondOutput):
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
        # if(heroes[heroIndex]=='ana'):
        #      print("shape is: " + str(sess.run(tf.shape(predictor),feed_dict={X:np.full(fill_value=.999,shape=[3,18])})))
        #      print("shape2 is: " + str(
        #          sess.run(tf.shape(newOutput), feed_dict={X: np.full(fill_value=.999, shape=[3, 18])})))
        #         # init_new_vars_op = tf.initialize_variables([lastOutput])
        #         # sess.run(init_new_vars_op)


        keepProbs.append(keep_prob)
        heroInput.append(X)
        if (outputVec != None):
            outputVec = tf.concat([outputVec, predictor], 1)
            print("shape of output: " + str(tf.shape(outputVec)))
        else:
            outputVec = (predictor)
            print("shape of output2: " + str(tf.shape(outputVec)))
    return outputVec



def forwardprop(X, w_1, w_2, w_3, w_32,keep_prop=1.0, batch_size=128):
    """
    Forward-propagation.
    IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
    """
    resultMultiplier = tf.floor(tf.add(tf.reduce_mean(X,axis=1),1.0))
    h1 = tf.nn.relu(tf.matmul(X, w_1))  # The \sigma function
    h2 = tf.nn.relu(tf.matmul(h1, w_2))

    yhat = tf.matmul(h2, w_3)  # The \varphi function
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
    #     if (scatter == None):
    #         scatter = tf.scatter_nd(indices, updates, shape)
    #     else:
    #         scatter = tf.group([scatter,tf.scatter_nd(indices, updates, shape)],1)
    # print("building shape is " + str(tf.Session().run(yhat2, feed_dict={X:np.zeros([2,18])})))
    return tf.concat([tf.transpose(tf.multiply(tf.transpose(yhat2),resultMultiplier)),tf.transpose(tf.multiply(tf.transpose(additionOutput),resultMultiplier))],1)

main()


while True:

    try:
        port_num = 9204
        break
    except ValueError:
        pass

ThreadedServer('127.0.0.1',port_num).listen()