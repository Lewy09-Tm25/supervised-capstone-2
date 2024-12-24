# Automated Heterogeneous Database Integration

The repository contains the papers used for the survey and pilot experiements of automated integration of databases. While there have been advancements in database management, the knowledge discover is limited in scope due to the complexity of merging datasets that are heterogeneous in nature.

The folder `Barbella et al. replication` consists of the code for replicating the methodology of state-of-the-art research in this domain. [Paper](https://www.sciencedirect.com/science/article/pii/S0167865523000132?via%3Dihub).

The file `experiments and analysis` consists of general analysis and insights we obtained from the datasets.

The folder `papers` consists of the research papers that were surveyed in order to understand the overall idealogy behind SOTA research in this field.

The datasets on which the evaluation was conducted are as follows:
1. Heat Index\
   a. The Heat Vulnerability Index (HVI) dataset shows NYC neighborhoods whose residents are more at risk for dying during and immediately following extreme heat. It uses a statistical model to summarize the most important social and environmental factors that contribute to neighborhood heat risk.\
   b. Rows- 197, Columns- 6\
   c. [Source](https://a816-dohbesp.nyc.gov/IndicatorPublic/data-explorer/climate/?id=2411#display=summary)
2. Traffic\
   a. New York City Department of Transportation (NYC DOT) uses Automated Traffic Recorders (ATR) to collect traffic sample volume counts at bridge crossings and roadways.\
   b. Rows- 1712605, Columns- 14\
   c. [Source](https://data.cityofnewyork.us/Transportation/Automated-Traffic-Volume-Counts/7ym2-wayt/about_data)
3. Tree Census\
   a. This dataset includes a record for every tree in New York City and includes the tree's location by borough and latitude/longitude, species by Latin name and common names, size, health, and issues with the tree's roots, trunk, and branches.\
   b. Rows- 683788, Columns- 41\
   c. [Source](https://www.kaggle.com/datasets/nycparks/tree-census?select=new_york_tree_census_2015.csv)
