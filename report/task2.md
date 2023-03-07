# Task 2: finding stuff

## Observations of data
Multiple articles contain the carriage return "\r", pandas' csv reader interprets this as a newline and thus goes to the next row and breaks everything.  

Some columns are entirely comprised of missing values, some have mostly missing values and therefore aren't that useful for a model, and the more important fields (content, label) also occasionally contain missing values.  

Some articles start with one or more newlines or end with one or more newlines.  

## Considerations about data
We decided to remove all the timestamps since they have no predicting power, they only tell us when the data was scraped and processed.  
