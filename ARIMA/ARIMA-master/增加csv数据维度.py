


createNum = 2
sourceFilePath = "international-airline-passengers.csv"
targetFilePath = "international-airline-passengers-target.csv"
def fun():
    with open(sourceFilePath,"r") as sourceFile:
        lineList = sourceFile.readlines()
    newlineList = []
    for line in lineList:
        if "," in line:
            index = line.index(",")
            appendString = line[index:-1]*createNum
            newlineList.append(f"{line[:-1]}{appendString}\n")
        else:
            newlineList.append(line)

    with open(targetFilePath,"w") as targetFile:
        targetFile.writelines(newlineList)




fun()




"""












"""