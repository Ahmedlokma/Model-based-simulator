import re
import json
def main ():
    # db_entity_file = open('/Users/ahmedlokma/Desktop/user-simulator-master/data/multiwoz-master/data/multi-woz/trainmodelentity.json','rb')
    db_entity_file = open('/Users/ahmedlokma/Desktop/user-simulator-master/data/multiwoz-master/data/multi-woz/Guc_Dataset_Entity22.json','rb')
    dataList= json.load(db_entity_file)
    
    # j_String  = {"qId": "wqr000001", "entities": [["natalie portman", "PERSON"], ["natalie portman", "NP"], ["star wars", "NP"]]}
    # j_Object = json.dumps(j_String)
    temp = []
    count = 0
    temp1 = []
    count1 = 0
    for index in range(len(dataList)): # list of dict
     
     for key in dataList[index]:
       
        if(key == "entities") :
         j_Object = dataList[index][key]
        
         elemnt = j_Object[0][1]
         
          
         if elemnt not in temp :
          temp.insert(count,j_Object[0][1])  
          count = count + 1 
        #   print(len(temp))
         if(elemnt == "guc_guidelines"):
             if(j_Object[0][0] not in temp1):
              temp1.insert(count1,j_Object[0][0])
              count1 = count1 +1


    print(len(temp1))
  
    x=""

    # with open('/Users/ahmedlokma/Desktop/user-simulator-master/data/multiwoz-master/data/multi-woz/example.json', 'w') as f:
    with open('/Users/ahmedlokma/Desktop/user-simulator-master/data/multiwoz-master/data/multi-woz/Guc_Dataset_Entity_Sorted22.json', 'w') as f:
     jsonString = json.dumps(temp1)
 
     for item in jsonString:
         if item ==',':
             x = x + ","
             f.write("%s\n" % x) 
             x = ""
         elif item == ' ':
             x = x+""+item  
         elif item == '[' :
             f.write("%s\n" % item)
             continue 
         elif  item == ']':
            f.write("%s\n" % x)
            f.write("%s\n" % item)
            continue
         else :
             x = x +""+item
        
            

if __name__ == '__main__':
    main()    
