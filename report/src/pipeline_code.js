flowchart LR;
A("RAW");
C("TRAIN");
D("VAL");
E("TEST");
F("TF-IDF - 2048");
G("TF-IDF - 4096");
H("SVD - 256");
I("SVD - 384");
J("Logistic regression");
K("XGBoost");
L("Small DNN");
M("Large DNN");
N("CLEANING");
O("TOKENIZATION");
P("TEST-TRAIN-VAL SPLIT")

subgraph Treatment
    direction TB; 
    A --> N;
    N --> O;
end;
Treatment --> P;
P --> C;
P --> D;
P --> E;
C --> F;
C --> G;
F --> H;
H --> J;
G --> I;
I --> K;
I --> L;
G --> M;
L --> D;

