import math
import matplotlib.pyplot as plt

# Function to get net of input
def netx(inputNum, o):
    return ( (inputNum[0] * o[0]) + (inputNum[1] * o[1]) + (inputNum[2] * o[2] + (inputNum[3] * o[3])  ))

# Function to get sigmoid of input
def sigmoidx(x):
    return 1 / (1 + math.exp(-x))

# Function to get softmax of input
def softmaxx(net1, net2):
    netAddition = math.exp(net1) + math.exp(net2)
    return (math.exp(net1) / netAddition)

# Function to calculate a point for the learning curve
def trainingError(target1, target2, output1, output2):
    return ((((target1-output1)**2)+((target2-output2)**2))/2)

# Define weights
#w04, w14, w24, w34
weightA4 = [0.9, 0.74, 0.8, 0.35]
#w05, w15, w25, w35
weightA5 = [0.45, 0.13, 0.4, 0.97]
#w06, w16, w26, w36
weightA6 = [0.36, 0.68, 0.1, 0.96]
#w07, w47, w57, w67
weightA7 = [0.98, 0.35, 0.5, 0.9]
#w08, w48, w58, w68
weightA8 = [0.92, 0.8, 0.13, 0.8]

# Define learning rates
learningRate = 0.1

# Create empty arrays to be filled later
testLine = []
fileData = []
trainingErrorPoints = []
# Create variable to hold the sum of training errors for each epoch
trainingErrorSum = 0




# Fill array with training data

# numLines to store how many lines of data are in the file
numLines = 0

# Open training data file
file = open("data-CMP2020M-item1-train.txt", "r")
# Read every line from training data into an array
while True:
    
    #Split line into array
    fileLine = (file.readline())
    
    # If nothing on line, break loop
    if fileLine == '':
        break

    # Split fileLine do 
    splitLine = fileLine.split()    
    # x0, x1, x2, x3, target1, target 2
    line = [1, float(splitLine[0]), float(splitLine[1]), float(splitLine[2]), int(splitLine[3]), int(splitLine[4])]
    fileData.append(line)

    #fileData formatting:
    #fileData[line][xX or targetX]
    #e.g.
    #fileData[3][2] = line 4 x2
    #filedata[2][4] = line 3 target1

    # Increment numLines
    numLines += 1
    
# Close file after use
file.close()

# Run for given number of epochs
for epoch in range(1, 101):
    # Runs through forward and backward step for each line in the training file
    for i in range(0, numLines):
        # Forward step
        net4 = netx(weightA4, [fileData[i][0], fileData[i][1], fileData[i][2], fileData[i][3]])
        net5 = netx(weightA5, [fileData[i][0], fileData[i][1], fileData[i][2], fileData[i][3]])
        net6 = netx(weightA6, [fileData[i][0], fileData[i][1], fileData[i][2], fileData[i][3]])
        
        o4 = sigmoidx (net4)
        o5 = sigmoidx (net5)
        o6 = sigmoidx (net6)
        
        net7 = netx(weightA7, [fileData[i][0], o4, o5, o6])
        net8 = netx(weightA8, [fileData[i][0], o4, o5, o6])

        o7 = net7 
        o8 = net8

        # Backward step

        # Calculate errors

        #Output errors
        outputError7 = fileData[i][4] - o7
        outputError8 = fileData[i][5] - o8

        #Hidden errors
        # a7[1] is w47, a8[1] is w48
        hiddenError4 = o4 * (1 - o4) * ( (weightA7[1] * outputError7) + (weightA8[1] * outputError8))
        # a7[1] is w57, a8[1] is w58
        hiddenError5 = o5 * (1 - o5) * ( (weightA7[2] * outputError7) + (weightA8[2] * outputError8))
        # a7[1] is w67, a8[1] is w68
        hiddenError6 = o6 * (1 - o6) * ( (weightA7[3] * outputError7) + (weightA8[3] * outputError8))
        
        # Display Errors for debug purposes
        #print ("HiddenError4: " + str(hiddenError4) + "\nHiddenError5: " + str(hiddenError5)  + "\nHiddenError6: " + str(hiddenError6))
        #print ("OutputError1: " + str(outputError7) + "\nOutputError2: " + str(outputError8) )
        

        # Recalculate Output Values

        #weightUpdateX = learningRate * hiddenErrorX * weightX
        #newWeight = oldWeight + weightUpdate
        weightA4 = [ (weightA4[0] + (learningRate * hiddenError4 * fileData[i][0])),  (weightA4[1] + (learningRate * hiddenError4 * fileData[i][1])),  (weightA4[2] + (learningRate * hiddenError4 * fileData[i][2])),  (weightA4[3] + (learningRate * hiddenError4 * fileData[i][3]))]
        weightA5 = [ (weightA5[0] + (learningRate * hiddenError5 * fileData[i][0])),  (weightA5[1] + (learningRate * hiddenError5 * fileData[i][1])),  (weightA5[2] + (learningRate * hiddenError5 * fileData[i][2])),  (weightA5[3] + (learningRate * hiddenError5 * fileData[i][3]))]
        weightA6 = [ (weightA6[0] + (learningRate * hiddenError6 * fileData[i][0])),  (weightA6[1] + (learningRate * hiddenError6 * fileData[i][1])),  (weightA6[2] + (learningRate * hiddenError6 * fileData[i][2])),  (weightA6[3] + (learningRate * hiddenError6 * fileData[i][3]))]

        #weightUpdateX = learningRate * outputErrorx * (weightX or ox)
        #newWeight = oldWeight + weightUpdate  
        weightA7 = [ (weightA7[0] + (learningRate * outputError7 * fileData[i][0])), (weightA7[1] + (learningRate * outputError7 * o4)), (weightA7[2] + (learningRate * outputError7 * o5)), (weightA7[3] + (learningRate * outputError7 * o6))]
        weightA8 = [ (weightA8[0] + (learningRate * outputError8 * fileData[i][0])), (weightA8[1] + (learningRate * outputError8 * o4)), (weightA8[2] + (learningRate * outputError8 * o5)), (weightA8[3] + (learningRate * outputError8 * o6))]

        # Add training error of current step to total training error of given epoch
        trainingErrorSum += trainingError(fileData[i][4], fileData[i][5], o7, o8)

    # Append total sum of training errors for epoch to array
    trainingErrorPoints.append(trainingErrorSum)
    # Reset sum of traing errors for next epoch
    trainingErrorSum = 0


    # Display output values for debug purposes
    #print (str(epoch))
    #print ("w04: " + str(weightA4[0]) + "\nw14: " + str(weightA4[1])  + "\nw24: " + str(weightA4[2]) + "\nw34: " + str(weightA4[3]))
    #print ("w05: " + str(weightA5[0]) + "\nw15: " + str(weightA5[1])  + "\nw25: " + str(weightA5[2]) + "\nw35: " + str(weightA5[3]))
    #print ("w06: " + str(weightA6[0]) + "\nw16: " + str(weightA6[1])  + "\nw26: " + str(weightA6[2]) + "\nw36: " + str(weightA6[3]))
    #print ("w07: " + str(weightA7[0]) + "\nw47: " + str(weightA7[1])  + "\nw57: " + str(weightA7[2]) + "\nw67: " + str(weightA7[3]))
    #print ("w08: " + str(weightA8[0]) + "\nw48: " + str(weightA8[1])  + "\nw58: " + str(weightA8[2]) + "\nw68: " + str(weightA8[3]) + "\n\n\n")



# Fill array with test data

# Open test data file
testFile = open("data-CMP2020M-item1-test.txt", "r")
#Split line into array
testFileLine = (testFile.readline())
# Split fileLine do 
splitLine = testFileLine.split()    
# x0, x1, x2, x3
testLine = [1, float(splitLine[0]), float(splitLine[1]), float(splitLine[2])]
# Close file after use
testFile.close()

# Calculate output from test values
net4 = netx(weightA4, [testLine[0], testLine[1], testLine[2], testLine[3]])
net5 = netx(weightA5, [testLine[0], testLine[1], testLine[2], testLine[3]])
net6 = netx(weightA6, [testLine[0], testLine[1], testLine[2], testLine[3]])
o4 = sigmoidx (net4)
o5 = sigmoidx (net5)
o6 = sigmoidx (net6)
net7 = netx(weightA7, [testLine[0], o4, o5, o6])
net8 = netx(weightA8, [testLine[0], o4, o5, o6])
o7 = net7 
o8 = net8

# Display output of test values
print ("y1: " + str(net7))
print ("y2: " + str(net8))



print ("softmax y1/Probability distribution for y1: " + str(softmaxx(o7, o8)))
print ("softmax o8/Probability distribution for y1: " + str(softmaxx(o8, o7)))


# Take the training error points and plot it as a line graph.
plt.plot(trainingErrorPoints)
plt.show() 











