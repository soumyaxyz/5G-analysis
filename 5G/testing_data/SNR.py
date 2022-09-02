# from sys import argv


# def getArgs():
#     try:
#         inFile = argv[1]
#         outFile = argv[2]
#         srcPattern = argv[3]
#     except:
#         print("test_data_100.text output_file.txt search_pattern")
#         exit()
#     return inFile, outFile


# # def parsetxt(inFile):
# #     # fasta will be a list of dictionaries.
# #     txt = []
#     lines = open(inFile, "r").readlines()
    
# #     # for line in lines:
# #     #     if line.startswith("Rx Signal to Noise Ratio"):
           

# # print()

# mylines = []                              
# with open ('test_data_100.txt', 'rt') as myfile:  
#     for line in myfile:                   
#         mylines.append(line)              
#     for element in mylines:               
#         print(element, end='')   
import csv
import numpy as np


##### These vales need to be modified####
fhand = open('stat_1000PingPacket_25feet_45degree.txt')
output_file_name = "SNR_data_1000_25feet_45degree.csv"
length = 1000
#f = open('csv_file.csv', 'w')
# create the csv writer
#writer = csv.writer(f)
############################

number = np.arange(length)
print(length)
#print(number)
value = np.zeros((length,32))
#print(value)
i1 = 0
i2 = 0
i3 = 0
i4 = 0
i5 = 0
i6 = 0
i7 = 0
i8 = 0
i9 = 0
i10 = 0
i11 = 0
i12 = 0
i13 = 0
i14 = 0
i15 = 0
i16 = 0
i17 = 0
i18 = 0
i19 = 0
i20 = 0
i21 = 0
i22 = 0
i23 = 0
i24 = 0
i25 = 0
i26 = 0
i27 = 0
i28 = 0
i29 = 0
i30 = 0
i31 = 0
i32 = 0
i33 = 0
i34 = 0
i35 = 0
i36 = 0
i37 = 0


for i in range(0,length):
    #print(i)
    value[i,0]=i+1

for line in fhand:
    line = line.rstrip()
    #print("printing line")
    #print(line)
    if (line.find('Rx Power:') != -1) and (i1<length):
        #print("Rx Power")
        #print(line[10:15])
        #value[i1,1] = float(line[10:15])
        s=line[10:]
        a,b = s.split(' ', 1)
        value[i1,1] = float(a)
        #print("I am here")
        #print(a,b)
        i1 = i1+1
        continue


    #print(line)
    if (line.find('Rx Power Average:') != -1)and (i2<length):
        #print("Rx Power Average:")
        #print(line[18:24])
        #value[i2,2] = float(line[18:24])
        #print("printing i2",i2)
        #print(value[i2,1])
        s=line[18:]
        a,b = s.split(' ', 1)
        value[i2,2] = float(a)
        #print("I am here")
        #print(a,b)
        i2 = i2+1
        continue
    #print(line)      
    #      
    if (line.find('Rx Power OTA:') != -1) and (i3<length):
    #    print("Rx Power OTA:")
    #    print(line[14:20])
        #value[i3,3] = float(line[14:20])
    #    print("printing i3",i3)
    #    print(value[i3,3])
        s=line[14:]
        a,b = s.split(' ', 1)
        value[i3,3] = float(a)
        #print("I am here")
        #print(a,b)
        i3 = i3+1
        continue
    
    #print(i)

         
    if (line.find('Rx Power Average OTA:') != -1)and (i4<length):
    #   print("Rx Power Average OTA:")
    #   print(line[23:28])
       #value[i4,4] = float(line[23:28])
    #   print("printing i4",i4)
    #   print(value[i4,4])
        s=line[23:]
        a,b = s.split(' ', 1)
        value[i4,4] = float(a)
        #print("I am here")
        #print(a,b)
        i4 = i4+1
        continue
    
    #print(i)

       #      
    if (line.find('Rx Signal to Noise Ratio:') != -1)and (i5<length):
        #print("Rx Signal to Noise Ratio:")
        #print(line[26:31])
        #value[i5,5] = float(line[26:31])
        #print("printing i5",i5)
        #print(value[i5,0])
        #print(line)
        s=line[26:]
        a,b = s.split(' ', 1)
        value[i5,5] = float(a)
        #print("I am here")
        #print(a,b)
        i5 = i5+1
        continue
    
#     #print(i)

#          #      
    if (line.find('Rx Average Signal to Noise Ratio') != -1)and (i6<length):
         #print("Rx Average Signal to Noise Ratio:")
         #print(line[34:39])
         #value[i6,6] = float(line[34:39])
         #print("printing i6",i6)
         #print(value[i6,0])
        #print(line)
        s=line[34:]
        a,b = s.split(' ', 1)
        value[i6,6] = float(a)
        #print("I am here")
        #print(a,b)
        i6 = i6+1
        continue
    
#     #print(i)

#          #      
    if (line.find('Rx SNR over Gi64:') != -1) and (i7<length):
         #print("Rx SNR over Gi64:")
         #print(line[18:23])
         #value[i7,7] = float(line[18:23])
         #print("printing i7",i7)
         #print(value[i7,0])
        #print(line)
        s=line[18:]
        a,b = s.split(' ', 1)
        value[i7,7] = float(a)
        #print("I am here")
        #print(a,b)
        i7 = i7+1
        continue
    
#     #print(i)

#           #      
    if (line.find('Rx Average SNR over Gi64') != -1)and (i8<length):
        #  print("Rx Average SNR over Gi64")
        #  print(line[25:30])
        # value[i8,8] = float(line[25:30])
        #  print("printing i8",i8)
        #  print(value[i8,8])
        #print(line)
        s=line[25:]
        a,b = s.split(' ', 1)
        value[i8,8] = float(a)
        #print("I am here")
        #print(a,b)
        i8 = i8+1
        continue
    
#     #print(i)

    

#           #      
    if (line.find('Rx Packet Error Rate (PER)') != -1)and (i9<length):
        #  print("Rx Packet Error Rate (PER)")
        #  print(line[28:33])
        # value[i9,9] = float(line[28:33])
        #  print("printing i9",i9)
        #  print(value[i9,9])
        #print(line)
        s=line[28:]
        a,b = s.split(' ', 1)
        value[i9,9] = float(a)
        #print("I am here")
        #print(a,b)
        i9 = i9+1
        continue
    
#     #print(i)

#          #      
    if (line.find('Tx Missed Ack Rate') != -1)and (i10<length):
        #  print("Tx Missed Ack Rate")
        #  print(line[19:23])
        # value[i10,10] = float(line[19:23])
        #  print("printing i10",i10)
        #  print(value[i10,10])
        #print(line)
        s=line[19:]
        a,b = s.split(' ', 1)
        value[i10,10] = float(a)
        #print("I am here")
        #print(a,b)
        i10 = i10+1
        continue
    
#     #print(i)
#      #      
    if (line.find('Rx AGC attenuation:') != -1)and (i11<length):
        #  print("Rx AGC attenuation:")
        #  print(line[20:25])
        # value[i11,11] = float(line[20:25])
        #  print("printing i11",i11)
        #  print(value[i11,11])
        #print(line)
        s=line[20:]
        a,b = s.split(' ', 1)
        value[i11,11] = float(a)
        #print("I am here")
        #print(a,b)
        i11 = i11+1
        continue
    
#     #print(i)
#          #      
    if (line.find('Rx average AGC attenuation:') != -1)and (i12<length):
        #  print("Rx average AGC attenuation:")
        #  print(line[28:33])
        # value[i12,12] = float(line[28:33])
        #  print("printing i12",i12)
        #  print(value[i12,12])
        #print(line)
        s=line[28:]
        a,b = s.split(' ', 1)
        value[i12,12] = float(a)
        #print("I am here")
        #print(a,b)
        i12 = i12+1
        continue
    
#     #print(i)
#          #      
    if (line.find('Tx MCS:') != -1)and (i13<length):
        #  print("Tx MCS:")
        #  print(line[8:10])
        # value[i13,13] = float(line[8:10])
        #  print("printing i13",i13)
        #  print(value[i13,13])
        #print(line)
        s=line[8:]
        value[i13,13] = float(s)
        #print("I am here")
        i13 = i13+1
        continue

#     #print(i)
#          #      
    if (line.find('Rx MCS:') != -1)and (i14<length):
        #  print("Rx MCS:")
         #print(line)
         value[i14,14] = float(line[8:])
        #  print("printing i14",i14)
        #  print(value[i14,14])
         #print("I am here")
         i14 = i14+1
         continue    
#     #print(i)
#          #      
    if (line.find('Local Device Tx sector:') != -1)and (i15<length):
        # print("Tx Missed Ack Rate:")
        # print(line[44:27])
        value[i15,15] = float(line[24:])
        # print("printing i15",i15)
        # print(value[i15,0])
        i15 = i15+1
        continue
    
#     #print(i)
#          #      
    if (line.find('Local Device Rx sector:') != -1)and (i16<length):
        # print("Rx Missed Ack Rate:")
        # print(line[44:27])
        value[i16,16] = float(line[24:])
        #print("I am here")
        #print(value[i16,16])
        # print("printing i16",i16)
        # print(value[i16,16])
        i16 = i16+1
        continue
    
#     #print(i)
#          #      
    if (line.find('TXSS periods in which an SSW frame was received') != -1)and (i17<length):
        # print("TXSS periods in which an SSW frame was received")
        # print(line[47:51])
        #value[i17,17] = float(line[47:51])
        # print("printing i17",i17)
        # print(value[i17,17])
        #i17 = i17+1
        #print("Inactive time:")
        #print(line)
        #print(line[47:])
        s=line[47:]
        a,b = s.split(' ', 1)
        #print("a = ", a)
        #print("b = ", b)
        #print("c = ",c)
        #print(b)
        value[i17,17] = float(b)
        # print("printing i30",i30)
        # print(value[i30,30])
        i17 = i17+1
        continue

    
#     #print(i)
#          #      
    if (line.find('RXSS periods in which an SSW frame was received') != -1)and (i18<length):
        s=line[47:]
        a,b = s.split(' ', 1)
        value[i18,18] = float(b)
        i18 = i18+1
        continue
    
#     #print(i)
#          #      
    if (line.find('Best TXSS sector last BTI') != -1)and (i19<length):
        # print("Best TXSS sector last BTI 30 - SNR 0.00 dB, Rx Power")
        # print(line[53:59])
        #print(line)
        words = line.split();
        #print(words[-2])
        value[i19,19] = float(line[53:59])
        # print("printing i19",i19)
        # print(value[i19,19])
        i19 = i19+1
        continue
    
#     #print(i)
#          #      
    if (line.find('Best RXSS sector last BTI') != -1)and (i20<length):
        # print("Best RXSS sector last BTI 0 - SNR 0.00 dB, Rx Power")
        # print(line[52:59])
        #print(line)
        words = line.split();
        #print(words[-2])
        value[i20,20] = float(line[52:59])
        # print("printing i20",i20)
        # print(value[i20,20])
        i20 = i20+1
        continue
    
#     #print(i)
#          #      
    if (line.find('Ethernet Packets Sent Successfully:') != -1)and (i21<length):
        # print("Ethernet Packets Sent Successfully:")
        # print(line[36:41])
        value[i21,21] = float(line[36:])
        # print("printing i21",i21)
        # print(value[i21,21])
        i21 = i21+1
        continue
    
    #print(i)
#          #      
    if (line.find('Bytes Sent successfully:') != -1)and (i22<length):
        # print("Bytes Sent successfully:")
        # print(line[25:33])
        value[i22,22] = float(line[25:])
        # print("printing i22",i22)
        # print(value[i22,22])
        i22 = i22+1
        continue
    
#     #print(i)
         #      
    if (line.find('Bytes Transmitted (inc. retransm.):') != -1)and (i23<length):
        # print("Bytes Transmitted (inc. retransm.):")
        # print(line[36:42])
        value[i23,23] = float(line[36:])
        # print("printing i23",i23)
        # print(value[i23,23])
        i23 = i23+1
        continue
    
#     #print(i)
#          #      
    if (line.find('Bytes Transmitted inc. retransm., MAC framing:') != -1)and (i24<length):
        # print("Bytes Transmitted inc. retransm., MAC framing:")
        # print(line[46:55])
        value[i24,24] = float(line[46:])
        # print("printing i24",i24)
        # print(value[i24,24])
        i24 = i24+1
        continue
    
#     #print(i)
#          #      
    if (line.find('Announce frames sent:') != -1)and (i25<length):
        # print("Announce frames sent:")
        # print(line[22:28])
        value[i25,25] = float(line[22:])
        # print("printing i25",i25)
        # print(value[i25,25])
        i25 = i25+1
        continue
    
#          #      
    if (line.find('BI where announce frame was acked:') != -1)and (i26<length):
        # print("BI where announce frame was acked:")
        # print(line[35:41])
        #value[i26,26] = float(line[35:])
        # print("printing i26",i26)
        # print(value[i26,26])
        #print(line)
        s=line[35:]
        value[i26,26] = float(s)
        #print("I am here")
        #print(s)
        i26 = i26+1
        continue
    
#     #print(i)
#          #      
    if (line.find('BI where announce frame was not acked:') != -1)and (i27<length):
        # print("BI where announce frame was acked:")
        # print(line[37:38])
        #value[i27,27] = float(line[38:39])
        # print("printing i27",i27)
        # print(value[i27,27])
        #i27 = i27+1
        #print(line)
        #print(line[38:])
        s=line[38:]
        a,b = s.split(' ', 1)
        #print("a = ", a)
        #print("b = ", b)
        #print("c = ",c)
        #print(b)
        value[i27,27] = float(b)
        # print("printing i30",i30)
        # print(value[i30,30])
        i27 = i27+1
        
        continue
    
#     #print(i)
#          #      
    if (line.find('MPDUs received') != -1)and (i28<length):
        # print("MPDUs received")
        # print(line[17:23])
        value[i28,28] = float(line[17:])
        #print("printing i28",i28)
        #print(value[i28,28])
        i28 = i28+1
        continue
    
#     #print(i)
#          #      
    if (line.find('MPDUs transmitted') != -1)and (i29<length):
        # print("MPDUs transmitted")
        # print(line[18:23])
        value[i29,29] = float(line[18:])
        #print("printing i29",i29)
        #print(value[i29,29])
        i29 = i29+1
        continue
    
#     #print(i)
#          #      
    if (line.find('Inactive time:') != -1)and (i30<length):
        #print("Inactive time:")
        #print(line)
        #print(line[14:20])
        s=line[14:20]
        a,b,c = s.split(' ', 2)
        #print("a = ", a)
        #print("b = ", b)
        #print("c = ",c)
        #print(b)
        value[i30,30] = float(b)
        # print("printing i30",i30)
        # print(value[i30,30])
        i30 = i30+1
        continue
    
# #     #print(i)
# #          #      
# #     if line.find('Tx Missed Ack Rate  :') != -1:
# #         print("Tx Missed Ack Rate  :")
# #         print(line[14:20])
# #         value[i31,31] = float(line[20:24])
# #         print("printing i31",i31)
# #         print(value[i31,31])
# #         i31 = i31+1
# #         continue
    
# #     #print(i)
#      #      
    # if line.find('SP blocks 0 duration') != -1:
    #     print("SP blocks 0 duration")
    #     print(line[:23])
    #     value[i31,31] = float(line[:23])
    #     print("printing i31",i31)
    #     print(value[i31,31])
    #     i31 = i31+1
    #     continue

# # #print(i)
# #      #      
#     if line.find('CBAP blocks 0 duration') != -1:
#         print("CBAP blocks 0 duration")
#         print(line[24:25])
#         value[i32,32] = float(line[24:25])
#         print("printing i32",i32)
#         print(value[i32,32])
#         i32 = i32+1
#         continue

# #print(i)
# #      #      
#     if line.find('Tx Missed Ack Rate  :') != -1:
#         print("Tx Missed Ack Rate  :")
#         print(line[14:20])
#         value[i34,34] = float(line[20:24])
#         print("printing i34",i34)
#         print(value[i34,34])
#         i34 = i34+1
#         continue

# #print(i)
#      #      
#     if line.find('Tx Missed Ack Rate  :') != -1:
#         print("Tx Missed Ack Rate  :")
#         print(line[14:20])
#         value[i33,33] = float(line[20:24])
#         print("printing i33",i33)
#         print(value[i33,33])
#         i33 = i33+1
#         continue

# # #print(i)
# #      #      
#     if line.find('Tx Missed Ack Rate  :') != -1:
#         print("Tx Missed Ack Rate  :")
#         print(line[14:20])
#         value[i34,34] = float(line[20:24])
#         print("printing i34",i34)
#         print(value[i34,34])
#         i34 = i34+1
#         continue

# #print(i)
  
     #      
    if (line.find('BH2 temperature:') != -1)and (i31<length):
        #print("BH2 temperature:")
        #print(line[17:19])
        #value[i31,31] = float(line[17:])
        # print("printing i33",i33)
        # print(value[i33,33])
        s=line[17:]
        a,b = s.split(' ', 1)
        #print("I am here")
        #print(a)
        value[i31,31] = float(a)
        i31 = i31+1
        continue


    



    


#     # write a row to the csv file
#     #writer.writerow(line)
#     # close the file 
# #f.close()





test_100 = open(output_file_name,"w+")  
mywriter = csv.writer(test_100, delimiter=',')            
for j in range(0, length):
    #test_100.write("%d  " % number[j])   
    #print(value[j]) 
    #test_100.write( value[j])
    mywriter.writerow(value[j])
    #test_100.write("\n")


        
