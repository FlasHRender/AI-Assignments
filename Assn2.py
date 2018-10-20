from DecisionTree import DecisionTree
import time
from multiprocessing import Pool


def func(skp):
    Des = DecisionTree("datasets\\Sensorless_drive_diagnosis.csv")
    # Des = DecisionTree("datasets\\sensor_truncated.csv")
    Des.Config(binPercent=skp[0], maxDepth=skp[1])

    # Training times
    trSt = time.time()
    Des.Train()
    trEnd = time.time()
    evlTime = trEnd-trSt

    # Testing times
    tstSt = time.time()
    res = Des.Test(Des.rootData[:, :-1])
    tstEnd = time.time()
    tstTime = tstEnd - tstSt

    # Accuracy
    correct = 0
    wrong = 0
    for i in range(len(res)):
        if res[i] == Des.rootData[:, -1][i]:
            correct += 1
        else:
            wrong += 1
    accr = correct/(correct+wrong)*100

    # Sort and check times
    sortTime = sum(Des.srtTimes)
    chkTime = sum(Des.chkTimes)
    totTime = sortTime + chkTime

    # Time errors
    delta = totTime - evlTime

    print(  "Skips:"        + str(skp[0]) +
          "  MaxDepth:"     + str(skp[1]) +
          "  Accu:"         + str.format("{0:5.3f}" ,  accr) +
          "  TotTime:"      + str.format("{0:9.3f}" ,  Des.TrainTime) +
          "  (+/-)"         + str.format("{0: 6.3f}",  delta) +
          "  Sort %:"       + str.format("{0:7.3f}" ,  (sortTime/evlTime)*100) +
          "  Entr Time %:"  + str.format("{0:7.3f}" ,  (chkTime/evlTime)*100) +
          "  Test Time:"    + str.format("{0:6.3f}" ,  tstTime))

if __name__ == "__main__":
    func([3.5, 30])
