from operator import index
import re
import json
db_entity_file2 = open('/Users/ahmedlokma/Desktop/user-simulator-master/data/multiwoz-master/data/multi-woz/trainmodelentity.json','rb')
dataList2= json.load(db_entity_file2)
db_entity_file = open('/Users/ahmedlokma/Desktop/user-simulator-master/data/multiwoz-master/data/multi-woz/trainmodel.json','rb')
dataList= json.load(db_entity_file)
j_question_no =""
f = False  
entity_temp = []
question_temp = []
answer_temp = []
z = 0


for index in range(len(dataList)): # list of dict
    for key in dataList[index]:
        if(key == "qId"):
         j_question_no = dataList[index][key]
         j_entity = dataList2[index]["entities"]
         j_answer = dataList[index]["answers"]
         j_question = dataList[index]["qText"]

         question_temp.insert(index,j_question)
         answer_temp.insert(index,j_answer)
         entity_temp.insert(index,j_entity)
 
jsonentity = json.dumps(entity_temp)
jsonquestion = json.dumps(question_temp)
jsonanswer = json.dumps(answer_temp)
print(jsonquestion)
with open('/Users/ahmedlokma/Desktop/user-simulator-master/data/multiwoz-master/data/multi-woz/example3.json', 'w') as f:
 x =""
 x1 = ""
 x2 = ""
 counter = 0
 secondloop = False
 thirdloop = False 
 last = False 
 f.write("[")
 f.write("\n") 
 f.write("{")
 f.write("\n") 
#  for item in jsonquestion:

