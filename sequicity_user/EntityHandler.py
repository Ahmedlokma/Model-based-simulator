import json
import pickle
# db_entity_file = open('/Users/ahmedlokma/Desktop/user-simulator-master/data/multiwoz-master/data/multi-woz/trainmodelentity.json','rb')
db_entity_file = open('/Users/ahmedlokma/Desktop/user-simulator-master/data/multiwoz-master/data/multi-woz/Guc_Dataset_Entity22.json','rb')
dataList= json.load(db_entity_file)
entity_temp = []
for index in range(len(dataList)): # list of dict
    for key in dataList[index]:
         j_entity = dataList[index]["entities"]
         entity_temp.insert(index,j_entity)
 
jsonentity = json.dumps(entity_temp)
x2 = ""
print(jsonentity)
# with open('/Users/ahmedlokma/Desktop/user-simulator-master/data/multiwoz-master/data/multi-woz/goal.json', 'w') as f:
with open('/Users/ahmedlokma/Desktop/user-simulator-master/data/multiwoz-master/data/multi-woz/Guc_Dataset2_Goal.json', 'w') as f:
     f.write("[")
     f.write("\n") 
     for item3 in jsonentity:
                if item3 == '[' and x2 !="" and  x2[-1] == '[' :
                    continue
                if item3 == '[' :
                    x2 = x2+ "" + '['
                    continue 
                elif item3 == ']' and x2 != "":
                    x2 = x2+ "" + ']'
                    f.write(x2)
                    x2 = ""
                elif item3 == ']':
                    continue 
                else :
                    x2 = x2 + "" + item3 
     f.write("\n")
     f.write("]")                   


                 
