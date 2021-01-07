preprocessing tools is a collection from the moses project:  https://github.com/moses-smt/mosesdecoder  
dexml comes from the europarl website:  http://www.statmt.org/europarl/



Structure:
```
[Working Directory]  
.
|  
|  
+----[Model name] 
|    |  
|    |  copy here the content of the repo:  
|    |  
|    +----[preprocessing-tools] 
|    |    | 
|    |    |
|    |    +----[non breaking prefixes]  
|    |         
|    |         
|    |    
|    +----[easymt] 
|         |  
|         +----easymt.py  
|         |
|         +----parameters.json  
|  
+----[texts]  
     |  
     +----save here all datasets with format filename.language  
          es. newstest2009.en, newstest2009.it  
   ```
