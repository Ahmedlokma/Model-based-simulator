from operator import contains, index
import re
import json
import time 
# db_entity_file2 = open('/Users/ahmedlokma/Desktop/user-simulator-master/data/multiwoz-master/data/multi-woz/trainmodelentity.json','rb')
db_entity_file2 = open('/Users/ahmedlokma/Desktop/user-simulator-master/data/multiwoz-master/data/multi-woz/Guc_Dataset_Entity22.json','rb')
dataList2= json.load(db_entity_file2)
# db_entity_file = open('/Users/ahmedlokma/Desktop/user-simulator-master/data/multiwoz-master/data/multi-woz/trainmodel.json','rb')
db_entity_file = open('/Users/ahmedlokma/Desktop/user-simulator-master/data/multiwoz-master/data/multi-woz/Guc_Dataset22.json','rb')
dataList= json.load(db_entity_file)
j_question_no =""
f = False  
entity_temp = []
question_temp = []
answer_temp = []
z = 0
# for index2 in range(len(dataList2)):
#  for count in range(len(dataList2[index2])) :
#          for item in (dataList2[index2]["entities"] ) :

#              if(item[1]=="NP"):
#                  print(item)
#                  dataList2[index2]["entities"].remove(item)


for index in range(len(dataList)): # list of dict
    for key in dataList[index]:
        if(key == "qId"):
         j_question_no = dataList[index][key]
         counter  = 0 
        # for count in range(len(dataList2[index])) :
        #  for item in (dataList2[index]["entities"] ) :
        #      print("thereee")
        #      if(item[1]=="NP"):
        #          print(item)
        #          dataList2[index]["entities"].remove(item)

         j_entity = dataList2[index]["entities"] 
         j_answer = dataList[index]["answers"]
         j_question = dataList[index]["qText"]

         question_temp.insert(index,j_question)
         answer_temp.insert(index,j_answer)
         entity_temp.insert(index,j_entity)
 
jsonentity = json.dumps(entity_temp)
jsonquestion = json.dumps(question_temp)
jsonanswer = json.dumps(answer_temp)


# print(jsonentity)
# print(len(jsonentity))
# print(entity_temp[0][0])

# with open('/Users/ahmedlokma/Desktop/user-simulator-master/data/multiwoz-master/data/multi-woz/example2.json', 'w') as f:
with open('/Users/ahmedlokma/Desktop/user-simulator-master/data/multiwoz-master/data/multi-woz/Guc_Dataset_Dialogue_act22.json', 'w') as f:
   

 x =""
 x1 = ""
 x2 = ""
 counter = 0
 secondloop = False
 thirdloop = False 
 last = False
 jsonentity = json.dumps(entity_temp) 
#  print(jsonanswer)
 f.write("[")
 f.write("\n") 
 f.write("{")
 f.write("\n") 

 for item in jsonquestion:

         if secondloop == True :
             secondloop = False 
             multiple = False 
             for item2 in jsonanswer:
              counter = counter + 1   
              if item2 == ']':
                 f.write('"B": { ')
                 f.write("\n")
                 f.write('"sent":')
                 if(multiple == True):
                     f.write('"')
                     x1 = x1.replace('"',"")
                    # f.write("[") 
                    
                #  x1 = x1.replace('"',"")
                 f.write(x1)
                 if(multiple == True):
                     f.write('"')
                    # f.write("]") 
                 f.write("\n") 
                 f.write("},")
                 f.write("\n")    
                 f.write('"turn" : 0')    
                 f.write("\n")  
                 f.write("}") 
                 f.write("\n")  
                 f.write("]")
                 f.write("\n")  
                 f.write("},")
                 f.write("\n")  
                 f.write("{")
                 f.write("\n") 
                 answer_temp.pop(0)
                 jsonanswer = json.dumps(answer_temp)
                 x1 = ""
                 break 

              elif item2 == '[':
                  continue 
              elif item2 == ',' and x1 != "":
                 x1 = x1 + "" + item2  
                 multiple = True   
              else :
                  x1 = x1 + "" + item2 
        
         if thirdloop == True :
             thirdloop = False
             first = True 
             for item3 in jsonentity:
                if first == True :
                    f.write('"slu" : [')
                    f.write("\n")
                    first = False 
                if item3 == '[' and x2 != ""  and x2[-1]== '[' :
                    continue
                elif item3 == ',' and x2[-1] == ']':
                    f.write("{")
                    f.write("\n")
                    f.write('"act": "inform"') 
                    f.write(",")
                    f.write("\n")
                    f.write('"slots": [') 
                    f.write("\n")
                    f.write(x2)
                    f.write("\n")
                    f.write("]")
                    f.write("\n")
                    f.write("},")
                    f.write("\n")
                    x2 = ""
                elif item3 == ']' and x2[-1] == ']':
                    f.write("{")
                    f.write("\n")
                    f.write('"act": "inform"') 
                    f.write(",")
                    f.write("\n")
                    f.write('"slots": [') 
                    f.write("\n")
                    f.write(x2)
                    x2 = ""
                    entity_temp.pop(0)
                    jsonentity = json.dumps(entity_temp) 
                    f.write("\n")
                    f.write("]")
                    f.write("\n")
                    f.write("}")
                    f.write("\n")
                    f.write("]")
                    f.write("\n")
                    f.write("},")
                    f.write("\n")
                    secondloop = True 
                    break 
                else :
                    x2 = x2 + "" + item3 

                    

             
        


         if item ==',' and  x[-1] == '"':
            #  print(x[-1])
            #  time.sleep(40)
             f.write(  '"dial" : [' ) 
             f.write("\n")
             f.write("{")
             f.write("\n")
             f.write('"A": { ')
             f.write("\n")
             f.write('"transcript":')
             f.write(x)
             f.write(",")
             f.write("\n")
             f.write("\n")
             x = ""
             thirdloop = True
         elif item == ' ':
             x = x+""+item  
         elif item == '[' :
             continue 
         elif  item == ']':
             f.write(  '"dial" : [' ) 
             f.write("\n")
             f.write("{")
             f.write("\n")
             f.write('"A": { ')
             f.write("\n")
             f.write('"transcript":')
             f.write(x)
             f.write(",")
             f.write("\n")
             f.write("\n")
             x = ""
             thirdloop = True
         else :
             x = x +""+item   



 if thirdloop == True :
             thirdloop = False
             first = True 
             for item3 in jsonentity:
                if first == True :
                    f.write('"slu" : [')
                    f.write("\n")
                    first = False 
                if item3 == '[' and x2 != ""  and x2[-1]== '[' :
                    continue
                elif item3 == ',' and x2[-1] == ']':
                    f.write("{")
                    f.write("\n")
                    f.write('"act": "inform"') 
                    f.write(",")
                    f.write("\n")
                    f.write('"slots": [') 
                    f.write("\n")
                    f.write(x2)
                    f.write("\n")
                    f.write("]")
                    f.write("\n")
                    f.write("},")
                    f.write("\n")
                    x2 = ""
                elif item3 == ']' and x2[-1] == ']':
                    f.write("{")
                    f.write("\n")
                    f.write('"act": "inform"') 
                    f.write(",")
                    f.write("\n")
                    f.write('"slots": [') 
                    f.write("\n")
                    f.write(x2)
                    x2 = ""
                    entity_temp.pop(0)
                    jsonentity = json.dumps(entity_temp) 
                    f.write("\n")
                    f.write("]")
                    f.write("\n")
                    f.write("}")
                    f.write("\n")
                    f.write("]")
                    f.write("\n")
                    f.write("},")
                    f.write("\n")
                    secondloop = True 
                    break 
                else :
                    x2 = x2 + "" + item3    
 if secondloop == True :
             secondloop = False 
             multiple = False 
             for item2 in jsonanswer:
              counter = counter + 1   
              if item2 == ']':
                 f.write('"B": { ')
                 f.write("\n")
                 f.write('"sent":')
                 if(multiple == True):
                    
                    # f.write("[") 
                    f.write('"')

                 x1 = x1.replace('"',"")
                 f.write(x1)

                 
                 if(multiple == True):
                    f.write('"')
                    # f.write("]")
                    # x1.replace(""," ") 
                 f.write("\n") 
                 f.write("},")
                 f.write("\n")    
                 f.write('"turn" : 0')    
                 f.write("\n")  
                 f.write("}") 
                 f.write("\n")  
                 f.write("]")
                 f.write("\n")  
                 f.write("}")
                 f.write("\n")  
                 f.write("\n") 
                 answer_temp.pop(0)
                 jsonanswer = json.dumps(answer_temp)
                 x1 = ""
                 break 

              elif item2 == '[':
                  continue 
              elif item2 == ',' and x1 != "":
                 x1 = x1 + "" + item2  
                 multiple = True   
              else :
                  x1 = x1 + "" + item2                      
 f.write("]")
#  print(len(answer_temp))
#  print(len(entity_temp))
#  print(len(question_temp))

  
          


         