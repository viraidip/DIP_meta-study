'''
    General functions and global parameters, that are used in different scripts
'''
import os
import random
import json

import numpy as np
import pandas as pd
import seaborn as sns

from typing import Tuple
from Bio import SeqIO


# load config and assign values to global variables
DATAPATH = json.load(open("../../.config.json"))["DATAPATH"]
RESULTSPATH = json.load(open("../../.config.json"))["RESULTSPATH"]

# segments, nuclotides, and strains
CMAP = "Accent"
CUTOFF = 15
SEGMENTS = list(["PB2", "PB1", "PA", "HA", "NP", "NA", "M", "NS"])
NUCLEOTIDES = dict({"A": "Adenine", "C": "Cytosin", "G": "Guanine", "U": "Uracil"})
STRAINS = dict({
    "Cal07": "A/California/07/2009",
    "NC": "A/New Caledonia/20-JY2/1999",
    "Perth": "A/Perth/16/2009",
    "BLEE": "B/Lee/1940",
    "PR8": "A/Puerto Rico/8/1934",
    "WSN": "A/WSN/1933"
})

DATASET_STRAIN_DICT = dict({
    # H1N1
    "Alnaji2021": "PR8",
    "Pelz2021": "PR8",
    "Wang2023": "PR8",
    "Wang2020": "PR8",
 #   "Kupke2020": "PR8",
    "EBI2020": "PR8",
 #   "Vasilijevic2017 swine": "swine",
  #  "Vasilijevic2017 Cal09": "Cal09",
    "Alnaji2019_Cal07": "Cal07",
    "Alnaji2019_Cal07_time": "Cal07",
    "Alnaji2019_NC" : "NC",
    "Mendes2021": "WSN",
    # H3N2
    "Alnaji2019_Perth": "Perth",
    "WRA2021_A": "Connecticut",
    "Rattanaburi2022_H3N2": "Thailand",
    # H5N1
    "Penn2022": "Turkey",
    # H7N9
    "Lui2019": "Anhui",
    # B 
    "Alnaji2019_BLEE": "BLEE",
    "WRA2021_B": "Victoria",
    "Sheng2018": "Brisbane",
    # n.a.
    "Greninger_2_2023": "Greninger_cons",
    "Lauring2019": "BLEE",

    # "Southgate2019": "Yamagata"
})

ACCNUMDICT = dict({
    "Wang2023": dict({
        "SRR16770171" : dict({"IFNAR": "1", "IFNLR": "0", "Replicate": "1"}),
        "SRR16770172" : dict({"IFNAR": "1", "IFNLR": "0", "Replicate": "1"}),
        "SRR16770173" : dict({"IFNAR": "1", "IFNLR": "0", "Replicate": "1"}),
        "SRR16770174" : dict({"IFNAR": "1", "IFNLR": "0", "Replicate": "1"}),
        "SRR16770175" : dict({"IFNAR": "1", "IFNLR": "0", "Replicate": "1"}),
        "SRR16770181" : dict({"IFNAR": "0", "IFNLR": "1", "Replicate": "1"}),
        "SRR16770182" : dict({"IFNAR": "0", "IFNLR": "1", "Replicate": "1"}),
        "SRR16770183" : dict({"IFNAR": "0", "IFNLR": "1", "Replicate": "1"}),
        "SRR16770184" : dict({"IFNAR": "0", "IFNLR": "1", "Replicate": "1"}),
        "SRR16770185" : dict({"IFNAR": "0", "IFNLR": "1", "Replicate": "1"}),
        "SRR16770186" : dict({"IFNAR": "0", "IFNLR": "1", "Replicate": "1"}),
        "SRR16770191" : dict({"IFNAR": "1", "IFNLR": "1", "Replicate": "1"}),
        "SRR16770192" : dict({"IFNAR": "1", "IFNLR": "1", "Replicate": "1"}),
        "SRR16770193" : dict({"IFNAR": "1", "IFNLR": "1", "Replicate": "1"}),
        "SRR16770197" : dict({"IFNAR": "1", "IFNLR": "0", "Replicate": "2"}),
        "SRR16770198" : dict({"IFNAR": "1", "IFNLR": "0", "Replicate": "2"}),
        "SRR16770201" : dict({"IFNAR": "1", "IFNLR": "0", "Replicate": "2"}),
        "SRR16770200" : dict({"IFNAR": "1", "IFNLR": "0", "Replicate": "2"}),
        "SRR16770199" : dict({"IFNAR": "1", "IFNLR": "0", "Replicate": "2"}),
        "SRR16770207" : dict({"IFNAR": "0", "IFNLR": "1", "Replicate": "2"}),
        "SRR16770208" : dict({"IFNAR": "0", "IFNLR": "1", "Replicate": "2"}),
        "SRR16770209" : dict({"IFNAR": "0", "IFNLR": "1", "Replicate": "2"}),
        "SRR16770210" : dict({"IFNAR": "0", "IFNLR": "1", "Replicate": "2"}),
        "SRR16770211" : dict({"IFNAR": "0", "IFNLR": "1", "Replicate": "2"}),
        "SRR16770212" : dict({"IFNAR": "0", "IFNLR": "1", "Replicate": "2"}),
        "SRR16770219" : dict({"IFNAR": "1", "IFNLR": "1", "Replicate": "2"}),
        "SRR16770218" : dict({"IFNAR": "1", "IFNLR": "1", "Replicate": "2"}),
        "SRR16770217" : dict({"IFNAR": "1", "IFNLR": "1", "Replicate": "2"})
    }),
    "Wang2020": dict({
        "SRR7722028" : dict({"Cell": "A549", "Time": "6", "Replicate": "1"}),
        "SRR7722030" : dict({"Cell": "A549", "Time": "12", "Replicate": "1"}),
        "SRR7722032" : dict({"Cell": "A549", "Time": "24", "Replicate": "1"}),
        "SRR7722029" : dict({"Cell": "A549", "Time": "6", "Replicate": "2"}),
        "SRR7722031" : dict({"Cell": "A549", "Time": "12", "Replicate": "2"}),
        "SRR7722033" : dict({"Cell": "A549", "Time": "24", "Replicate": "2"}),

        "SRR7722036" : dict({"Cell": "HBEpC", "Time": "6", "Replicate": "1"}),
        "SRR7722038" : dict({"Cell": "HBEpC", "Time": "12", "Replicate": "1"}),
        "SRR7722040" : dict({"Cell": "HBEpC", "Time": "24", "Replicate": "1"}),
        "SRR7722037" : dict({"Cell": "HBEpC", "Time": "6", "Replicate": "2"}),
        "SRR7722039" : dict({"Cell": "HBEpC", "Time": "12", "Replicate": "2"}),
        "SRR7722041" : dict({"Cell": "HBEpC", "Time": "24", "Replicate": "2"})
    }),
    "Mendes2021": dict({
        "SRR15720520": dict({"Status": "enriched", "Virus": "1", "Replicate": "1"}),
        "SRR15720521": dict({"Status": "enriched", "Virus": "1", "Replicate": "2"}),
        "SRR15720522": dict({"Status": "enriched", "Virus": "2", "Replicate": "1"}),
        "SRR15720523": dict({"Status": "enriched", "Virus": "2", "Replicate": "2"}),
        "SRR15720524": dict({"Status": "depleted", "Virus": "1", "Replicate": "1"}),
        "SRR15720525": dict({"Status": "depleted", "Virus": "1", "Replicate": "2"}),
        "SRR15720526": dict({"Status": "depleted", "Virus": "2", "Replicate": "1"}),
        "SRR15720527": dict({"Status": "depleted", "Virus": "2", "Replicate": "2"})
    }),
    "Pelz2021": dict({
        "SRR15084925": dict({"Time": "seed"}),
        "SRR15084924": dict({"Time": "0.5dpi"}),
        "SRR15084913": dict({"Time": "1dpi"}),
        "SRR15084908": dict({"Time": "1.4dpi"}),
        "SRR15084907": dict({"Time": "3.5dpi"}),
        "SRR15084906": dict({"Time": "4dpi"}),
        "SRR15084905": dict({"Time": "4.5dpi"}),
        "SRR15084904": dict({"Time": "5dpi"}),
        "SRR15084903": dict({"Time": "5.5dpi"}),
        "SRR15084902": dict({"Time": "8dpi"}),
        "SRR15084923": dict({"Time": "9dpi"}),
        "SRR15084922": dict({"Time": "9.4dpi"}),
        "SRR15084921": dict({"Time": "12.4dpi"}),
        "SRR15084919": dict({"Time": "13dpi"}),
        "SRR15084918": dict({"Time": "13.5dpi"}),
        "SRR15084917": dict({"Time": "16dpi"}),
        "SRR15084916": dict({"Time": "17dpi"}),
        "SRR15084915": dict({"Time": "17.5dpi"}),
        "SRR15084914": dict({"Time": "18dpi"}),
        "SRR15084912": dict({"Time": "19.5dpi"}),
        "SRR15084911": dict({"Time": "20dpi"}),
        "SRR15084910": dict({"Time": "20.4dpi"}),
        "SRR15084909": dict({"Time": "21dpi"})
    }),
    "Alnaji2019_Cal07": dict({
        "SRR8754522": dict({"Lineage": "1", "Passage": "6"}),
        "SRR8754523": dict({"Lineage": "2", "Passage": "6"})
    }),
    "Alnaji2019_Cal07_time": dict({
        "SRR8754531": dict({"Lineage": "1", "Passage": "6"}),
        "SRR8754532": dict({"Lineage": "1", "Passage": "3"}),
        "SRR8754533": dict({"Lineage": "1", "Passage": "1"})
    }),
    "Alnaji2019_NC": dict({
        "SRR8754513": dict({"Lineage": "2", "Passage": "1"}),
        "SRR8754514": dict({"Lineage": "1", "Passage": "1"}),
   #     "SRR8754527": dict({"Lineage": "1", "Passage": "6"}),
    #    "SRR8754538": dict({"Lineage": "2", "Passage": "6"})
    }),
    "Alnaji2019_Perth": dict({
     #   "SRR8754517": dict({"Lineage": "2", "Passage": "8"}),
        "SRR8754524": dict({"Lineage": "1", "Passage": "4"}),
        "SRR8754525": dict({"Lineage": "2", "Passage": "4"}),
      #  "SRR8754526": dict({"Lineage": "1", "Passage": "8"})
    }),
    "Alnaji2019_BLEE": dict({
   #     "SRR8754507": dict({"Lineage": "1", "Passage": "8"}),
        "SRR8754508": dict({"Lineage": "2", "Passage": "7"}),
        "SRR8754509": dict({"Lineage": "1", "Passage": "7"}),
    #    "SRR8754516": dict({"Lineage": "2", "Passage": "8"})
    }),
    "Lui2019": dict({
        "SRR8949705": dict({}),
        "SRR8945328": dict({}),
    }),
    "Penn2022": dict({
        "ERR10231074": dict({"Time": "24hpi", "Mode": "High", "Lineage": "1"}),
        "ERR10231075": dict({"Time": "48hpi", "Mode": "High", "Lineage": "1"}),
        "ERR10231076": dict({"Time": "6hpi", "Mode": "High", "Lineage": "1"}),
        "ERR10231077": dict({"Time": "96hpi", "Mode": "High", "Lineage": "1"}),
        "ERR10231078": dict({"Time": "24hpi", "Mode": "High", "Lineage": "2"}),
        "ERR10231079": dict({"Time": "48hpi", "Mode": "High", "Lineage": "2"}),
        "ERR10231080": dict({"Time": "6hpi", "Mode": "High", "Lineage": "2"}),
        "ERR10231081": dict({"Time": "96hpi", "Mode": "High", "Lineage": "2"}),
        "ERR10231089": dict({"Time": "96hpi", "Mode": "Low", "Lineage": "2"}),
        "ERR10231082": dict({"Time": "24hpi", "Mode": "Low", "Lineage": "1"}),
        "ERR10231085": dict({"Time": "96hpi", "Mode": "Low", "Lineage": "1"}),
        "ERR10231083": dict({"Time": "48hpi", "Mode": "Low", "Lineage": "1"}),
        "ERR10231084": dict({"Time": "6hpi", "Mode": "Low", "Lineage": "1"}),
        "ERR10231086": dict({"Time": "24hpi", "Mode": "Low", "Lineage": "2"}),
        "ERR10231087": dict({"Time": "48hpi", "Mode": "Low", "Lineage": "2"}),
        "ERR10231088": dict({"Time": "6hpi", "Mode": "Low", "Lineage": "2"})
    }),
    "Alnaji2021": dict({
        "SRR14352106": dict({"Replicate": "C", "Time": "24hpi"}),
        "SRR14352107": dict({"Replicate": "B", "Time": "24hpi"}),
        "SRR14352108": dict({"Replicate": "A", "Time": "24hpi"}),
        "SRR14352109": dict({"Replicate": "C", "Time": "6hpi"}),
        "SRR14352110": dict({"Replicate": "B", "Time": "6hpi"}),
        "SRR14352111": dict({"Replicate": "A", "Time": "6hpi"}),
        "SRR14352112": dict({"Replicate": "C", "Time": "3hpi"}),
        "SRR14352113": dict({"Replicate": "X", "Time": "0hpi"}),
        "SRR14352116": dict({"Replicate": "B", "Time": "3hpi"}),
        "SRR14352117": dict({"Replicate": "A", "Time": "3hpi"})
    }),
    "Kupke2020": dict({
        "SRR10489473": dict({"Type": "singlecell"}),
        "SRR10489474": dict({"Type": "singlecell"}),
        "SRR10489475": dict({"Type": "singlecell"}),
        "SRR10489476": dict({"Type": "singlecell"}),
        "SRR10489477": dict({"Type": "singlecell"}),
        "SRR10489478": dict({"Type": "singlecell"}),
        "SRR10489479": dict({"Type": "singlecell"}),
        "SRR10489480": dict({"Type": "singlecell"}),
        "SRR10489481": dict({"Type": "singlecell"}),
        "SRR10489482": dict({"Type": "singlecell"}),
        "SRR10489483": dict({"Type": "singlecell"}),
        "SRR10489484": dict({"Type": "singlecell"}),
        "SRR10489485": dict({"Type": "singlecell"}),
        "SRR10489486": dict({"Type": "singlecell"}),
        "SRR10489487": dict({"Type": "singlecell"}),
        "SRR10489488": dict({"Type": "singlecell"}),
        "SRR10489489": dict({"Type": "singlecell"}),
        "SRR10489490": dict({"Type": "singlecell"}),
        "SRR10489491": dict({"Type": "singlecell"}),
        "SRR10489492": dict({"Type": "singlecell"}),
        "SRR10489493": dict({"Type": "singlecell"}),
        "SRR10489494": dict({"Type": "singlecell"}),
        "SRR10489495": dict({"Type": "singlecell"}),
        "SRR10489496": dict({"Type": "singlecell"}),
        "SRR10489497": dict({"Type": "singlecell"}),
        "SRR10489498": dict({"Type": "singlecell"}),
        "SRR10489499": dict({"Type": "singlecell"}),
        "SRR10489500": dict({"Type": "singlecell"}),
        "SRR10489501": dict({"Type": "singlecell"}),
        "SRR10489502": dict({"Type": "singlecell"}),
        "SRR10489503": dict({"Type": "singlecell"}),
        "SRR10489504": dict({"Type": "singlecell"}),
        "SRR10489505": dict({"Type": "singlecell"}),
        "SRR10489506": dict({"Type": "singlecell"}),
        "SRR10489507": dict({"Type": "singlecell"}),
        "SRR10489508": dict({"Type": "singlecell"}),
        "SRR10489509": dict({"Type": "singlecell"}),
        "SRR10489510": dict({"Type": "singlecell"}),
        "SRR10489511": dict({"Type": "singlecell"}),
        "SRR10489512": dict({"Type": "singlecell"}),
        "SRR10489513": dict({"Type": "singlecell"}),
        "SRR10489514": dict({"Type": "singlecell"}),
        "SRR10489515": dict({"Type": "singlecell"}),
        "SRR10489516": dict({"Type": "singlecell"}),
        "SRR10489517": dict({"Type": "singlecell"}),
        "SRR10489518": dict({"Type": "singlecell"}),
        "SRR10489519": dict({"Type": "singlecell"}),
        "SRR10489520": dict({"Type": "singlecell"}),
        "SRR10489521": dict({"Type": "singlecell"}),
        "SRR10489522": dict({"Type": "singlecell"}),
        "SRR10489523": dict({"Type": "singlecell"}),
        "SRR10489524": dict({"Type": "singlecell"}),
        "SRR10489525": dict({"Type": "singlecell"}),
        "SRR10489526": dict({"Type": "singlecell"}),
        "SRR10489527": dict({"Type": "singlecell"}),
        "SRR10489528": dict({"Type": "singlecell"}),
        "SRR10489529": dict({"Type": "singlecell"}),
        "SRR10489530": dict({"Type": "singlecell"}),
        "SRR10489531": dict({"Type": "singlecell"}),
        "SRR10489532": dict({"Type": "singlecell"}),
        "SRR10489533": dict({"Type": "singlecell"}),
        "SRR10489534": dict({"Type": "singlecell"}),
        "SRR10489535": dict({"Type": "singlecell"}),
        "SRR10489536": dict({"Type": "singlecell"}),
        "SRR10489537": dict({"Type": "singlecell"}),
        "SRR10489538": dict({"Type": "singlecell"}),
        "SRR10489539": dict({"Type": "singlecell"}),
        "SRR10489540": dict({"Type": "singlecell"}),
        "SRR10489541": dict({"Type": "singlecell"}),
        "SRR10489542": dict({"Type": "singlecell"}),
        "SRR10489543": dict({"Type": "singlecell"}),
        "SRR10489544": dict({"Type": "singlecell"}),
        "SRR10489545": dict({"Type": "singlecell"}),
        "SRR10489546": dict({"Type": "singlecell"}),
        "SRR10489547": dict({"Type": "singlecell"}),
        "SRR10489548": dict({"Type": "singlecell"}),
        "SRR10489549": dict({"Type": "singlecell"}),
        "SRR10489550": dict({"Type": "singlecell"}),
        "SRR10489551": dict({"Type": "singlecell"}),
        "SRR10489552": dict({"Type": "singlecell"}),
        "SRR10489553": dict({"Type": "singlecell"}),
        "SRR10489554": dict({"Type": "singlecell"}),
        "SRR10489555": dict({"Type": "singlecell"}),
        "SRR10489556": dict({"Type": "singlecell"}),
        "SRR10489557": dict({"Type": "singlecell"}),
        "SRR10489558": dict({"Type": "singlecell"}),
        "SRR10489559": dict({"Type": "singlecell"}),
        "SRR10489560": dict({"Type": "singlecell"}),
        "SRR10489561": dict({"Type": "singlecell"}),
        "SRR10489562": dict({"Type": "singlecell"}),
        "SRR10489563": dict({"Type": "singlecell"}),
        "SRR10489564": dict({"Type": "singlecell"}),
        "SRR10489565": dict({"Type": "singlecell"}),
        "SRR10489566": dict({"Type": "singlecell"}),
        "SRR10489567": dict({"Type": "singlecell"}),
        "SRR10489568": dict({"Type": "singlecell"}),
        "SRR10530642": dict({"Type": "pre"}),
        "SRR10530643": dict({"Type": "post"})
    }),
    "Sheng2018": dict({
        "SRR3211978": dict({}),
        "SRR3211980": dict({}),
        "SRR3211976": dict({}),
        "SRR3211977": dict({}),
        "SRR3211974": dict({}),
        "SRR3211975": dict({}),
        "SRR3211972": dict({})
    }),
    "Vasilijevic2017_swine": dict({
        "SRR3743505": dict({"Outcome": "Death", "Gender": "F"}),
        "SRR3743506": dict({"Outcome": "Death", "Gender": "F"}),
        "SRR3743507": dict({"Outcome": "Recovery", "Gender": "F"}),
        "SRR3743508": dict({"Outcome": "Recovery", "Gender": "F"}),
        "SRR3743509": dict({"Outcome": "Recovery", "Gender": "F"}),
        "SRR3743510": dict({"Outcome": "Recovery", "Gender": "M"}),
        "SRR3743512": dict({"Outcome": "Recovery", "Gender": "M"}),
        "SRR3743517": dict({"Outcome": "Mulitorgan Failure", "Gender": "M"}),
        "SRR3743524": dict({"Outcome": "Mulitorgan Failure", "Gender": "F"}),
        "SRR3743525": dict({"Outcome": "Mulitorgan Failure", "Gender": "F"}),
        "SRR3743526": dict({"Outcome": "Mulitorgan Failure", "Gender": "M"}),
        "SRR3743527": dict({"Outcome": "Death", "Gender": "F"}),
        "SRR3743529": dict({"Outcome": "Recovery", "Gender": "M"}),
        "SRR3743530": dict({"Outcome": "Recovery", "Gender": "F"}),
    }),
    "Vasilijevic2017_Cal09": dict({
        "SRR3743518": dict({}),
        "SRR3743519": dict({}),
        "SRR3743520": dict({}),
        "SRR3743521": dict({}),
        "SRR3743522": dict({}),
        "SRR3743523": dict({})
    }),
    "EBI2020": dict({
        "ERR4566024":  dict({"Cell": "A549"}),
        "ERR4566025":  dict({"Cell": "A549"}),
        "ERR4566028":  dict({"Cell": "HEK293FT"}),
        "ERR4566029":  dict({"Cell": "HEK293FT"}),
        "ERR4566032":  dict({"Cell": "MRC5"}),
        "ERR4566033":  dict({"Cell": "MRC5"}),
        "ERR4566036":  dict({"Cell": "WI38"}),
        "ERR4566037":  dict({"Cell": "WI38"})
    }),
    "Greninger_2_2023": dict({
        "SRR23634030": dict({"Strain": "B8E6"}),
        "SRR23634031": dict({"Strain": "B8B1"}),
        "SRR23634032": dict({"Strain": "B7N6"}),
        "SRR23634033": dict({"Strain": "B7C5"}),
        "SRR23634034": dict({"Strain": "B7H4"})
    }),
    "WRA2021_A": dict({
        "SRR15182178":  dict({}),
        "SRR15182177":  dict({}),
        "SRR15182176":  dict({}),
        "SRR15182175":  dict({}),
        "SRR15182174":  dict({}),
        "SRR15182173":  dict({}),
        "SRR15182172":  dict({}),
        "SRR15182171":  dict({})
    }),
    "WRA2021_B": dict({
        "SRR15183345":  dict({}),
        "SRR15183344":  dict({}),
        "SRR15183352":  dict({}),
        "SRR15183353":  dict({}),
        "SRR15196408":  dict({}),
        "SRR15196409":  dict({}),
        "SRR15196410":  dict({}),
        "SRR15196411":  dict({}),
        "SRR15196412":  dict({}),
        "SRR15196413":  dict({}),
        "SRR15196414":  dict({}),
        "SRR15196415":  dict({}),
        "SRR15196416":  dict({}),
        "SRR15196417":  dict({}),
        "SRR15196419":  dict({}),
        "SRR15196418":  dict({}),
        "SRR15196420":  dict({}),
        "SRR15196421":  dict({}),
        "SRR15196422":  dict({}),
        "SRR15196423":  dict({}),
        "SRR15196424":  dict({}),
        "SRR15196425":  dict({})
    }),
        "WRA2021_B_yamagata": dict({
        "SRR15183338":  dict({}),
        "SRR15183343":  dict({}),
        "SRR15183342":  dict({}),
        "SRR15183341":  dict({}),
        "SRR15183340":  dict({}),
        "SRR15183339":  dict({})
    }),
    "Rattanaburi2022_H3N2": dict({
        "SRR10256717":  dict({}),
        "SRR10256718":  dict({}),
        "SRR10256719":  dict({}),
        "SRR10256720":  dict({}),
        "SRR10256721":  dict({})
    }),
    "Lauring2019": dict({
        "SRR10013210": dict({}),
        "SRR10013205": dict({}),
        "SRR10013181": dict({}),
        "SRR10013264": dict({}),
        "SRR10013191": dict({}),
        "SRR10013228": dict({}),
        "SRR10013249": dict({}),
        "SRR10013260": dict({}),
        "SRR10013239": dict({}),
        "SRR10013211": dict({}),
        "SRR10013203": dict({}),
        "SRR10013255": dict({}),
        "SRR10013170": dict({}),
        "SRR10013175": dict({}),
        "SRR10013284": dict({})
    }),
    "Southgate2019": dict({
    })
})


SEGMENT_DICTS = dict({
    "PR8": dict({
        "AF389115.1": "PB2",
        "AF389116.1": "PB1",
        "AF389117.1": "PA",
        "AF389118.1": "HA",
        "AF389119.1": "NP",
        "AF389120.1": "NA",
        "AF389121.1": "M",
        "AF389122.1": "NS"
    }),
    "Cal07": dict({
        "CY121687.1": "PB2",
        "CY121686.1": "PB1",
        "CY121685.1": "PA",
        "CY121680.1": "HA",
        "CY121683.1": "NP",
        "CY121682.1": "NA",
        "CY121681.1": "M",
        "CY121684.1": "NS"
    }),
    "NC": dict({
        "CY147325.1": "PB2",
        "CY147324.1": "PB1",
        "CY147323.1": "PA",
        "CY147318.1": "HA",
        "CY147321.1": "NP",
        "CY147320.1": "NA",
        "CY147319.1": "M",
        "CY147322.1": "NS"
    }),
    "Perth": dict({
        "KJ609203.1": "PB2",
        "KJ609204.1": "PB1",
        "KJ609205.1": "PA",
        "KJ609206.1": "HA",
        "KJ609207.1": "NP",
        "KJ609208.1": "NA",
        "KJ609209.1": "M",
        "KJ609210.1": "NS"
    }),
    "BLEE": dict({
        "CY115118.1": "PB2",
        "CY115117.1": "PB1",
        "CY115116.1": "PA",
        "CY115111.1": "HA",
        "CY115114.1": "NP",
        "CY115113.1": "NA",
        "CY115112.1": "M",
        "CY115115.1": "NS"
    }),
    # Not used right now, rather use sequences provided by Mendes and Russell
    "___WSN": dict({
        "CY034139.1": "PB2",
        "CY034138.1": "PB1",
        "CY034137.1": "PA",
        "CY034132.1": "HA",
        "CY034135.1": "NP",
        "CY034134.1": "NA",
        "CY034133.1": "M",
        "CY034136.1": "NS"
    }),
    "WSN": dict({
        "PB2_vRNA": "PB2",
        "PB1_vRNA": "PB1",
        "PA_vRNA": "PA",
        "HA_vRNA": "HA",
        "NP_vRNA": "NP",
        "NA_vRNA": "NA",
        "M_vRNA": "M",
        "NS_vRNA": "NS"
    }),
    "Anhui": dict({
        "439504": "PB2",
        "439508": "PB1",
        "439503": "PA",
        "439507": "HA",
        "439505": "NP",
        "439509": "NA",
        "439506": "M",
        "439510": "NS"
    }),
    "Turkey": dict({
        "EF619975.1": "PB2",
        "EF619976.1": "PB1",
        "EF619979.1": "PA",
        "AF389118.1": "HA",
        "EF619977.1": "NP",
        "EF619973.1": "NA",
        "EF619978.1": "M",
        "EF619974.1": "NS"
    }),
    "Brisbane": dict({
        "CY115158.1": "PB2",
        "CY115157.1": "PB1",
        "CY115156.1": "PA",
        "CY115151.1": "HA",
        "CY115154.1": "NP",
        "CY115153.1": "NA",
        "CY115152.1": "M",
        "CY115155.1": "NS"
    }),
    "swine": dict({
        "KR701038.1": "PB2",
        "KR701039.1": "PB1",
        "KR701040.1": "PA",
        "KR701041.1": "HA",
        "KR701042.1": "NP",
        "KR701043.1": "NA",
        "KR701044.1": "M",
        "KR701045.1": "NS"
    }),
    "Cal09": dict({
        "JF915190.1": "PB2",
        "JF915189.1": "PB1",
        "JF915188.1": "PA",
        "JF915184.1": "HA",
        "JF915187.1": "NP",
        "JF915186.1": "NA",
        "JF915185.1": "M",
        "JF915191.1": "NS"
    }),
    "Greninger_2_2023": dict({
        "OQ535342.1": "PB2",
        "OQ535341.1": "PB1",
        "OQ535340.1": "PA",
        "OQ535335.1": "HA",
        "OQ535338.1": "NP",
        "OQ535337.1": "NA",
        "OQ535336.1": "M",
        "OQ535339.1": "NS",
        "OQ535366.1": "PB2",
        "OQ535365.1": "PB1",
        "OQ535364.1": "PA",
        "OQ535359.1": "HA",
        "OQ535362.1": "NP",
        "OQ535361.1": "NA",
        "OQ535360.1": "M",
        "OQ535363.1": "NS",
        "OQ535350.1": "PB2",
        "OQ535349.1": "PB1",
        "OQ535348.1": "PA",
        "OQ535343.1": "HA",
        "OQ535346.1": "NP",
        "OQ535345.1": "NA",
        "OQ535344.1": "M",
        "OQ535347.1": "NS",
        "OQ535358.1": "PB2",
        "OQ535357.1": "PB1",
        "OQ535356.1": "PA",
        "OQ535351.1": "HA",
        "OQ535354.1": "NP",
        "OQ535353.1": "NA",
        "OQ535352.1": "M",
        "OQ535355.1": "NS",
        "OQ535374.1": "PB2",
        "OQ535373.1": "PB1",
        "OQ535372.1": "PA",
        "OQ535367.1": "HA",
        "OQ535370.1": "NP",
        "OQ535369.1": "NA",
        "OQ535368.1": "M",
        "OQ535371.1": "NS"
    }),
    "Greninger_cons": dict({
        "PB2": "PB2",
        "PB1": "PB1",
        "PA": "PA",
        "HA": "HA",
        "NP": "NP",
        "NA": "NA",
        "M": "M",
        "NS": "NS"
    }),
    "Connecticut": dict({
        "KM654658.1": "PB2",
        "KM654706.1": "PB1",
        "KM654754.1": "PA",
        "KM654822.1": "HA",
        "KM654847.1": "NP",
        "KM654920.1": "NA",
        "KM654969.1": "M",
        "KM654612.1": "NS"
    }),
    "Victoria": dict({
        "CY018660.1": "PB2",
        "CY018659.1": "PB1",
        "CY018658.1": "PA",
        "CY018653.1": "HA",
        "CY018656.1": "NP",
        "CY018655.1": "NA",
        "CY018654.1": "M",
        "CY018657.1": "NS"
    }),
    "Thailand": dict({
        "KP335735.1": "PB2",
        "KP335793.1": "PB1",
        "KP335851.1": "PA",
        "KP335964.1": "HA",
        "KP336026.1": "NP",
        "KP336139.1": "NA",
        "KP336201.1": "M",
        "KP336259.1": "NS"
    }),
    "Yamagata":({
        
    })
})

# global colors for plotting
COLORS = dict({"A": "deepskyblue", "C": "gold", "G": "springgreen", "U": "salmon"})

# parameters for the sampling
QUANT = 0.1
N_SAMPLES = 2000

def load_single_dataset(exp: str, acc: str, segment_dict: dict)-> pd.DataFrame:
    '''
    
    '''
    path = os.path.join(DATAPATH, exp, f"{exp}_{acc}.csv")
    df = pd.read_csv(path,
                     dtype={"Segment": "string", "Start": "int64", "End": "int64", "NGS_read_count": "int64"},
                     na_values=["", "None"],
                     keep_default_na=False)
    df["Segment"] = df["Segment"].replace(segment_dict)

    return df

def load_dataset(dataset: str):
    '''
    
    '''
    acc_nums = ACCNUMDICT[dataset]
    strain = DATASET_STRAIN_DICT[dataset]
    dfs = list()
    for acc_num, meta in acc_nums.items():
        df = load_single_dataset(dataset, acc_num, SEGMENT_DICTS[strain])
        for key in meta.keys():
            df[key] = meta[key]
        dfs.append(df)
    concat_df = pd.concat(dfs)

    return concat_df

def load_all(dfnames: list, expected: str=False):
    '''
    
    '''
    dfs = list()
    expected_dfs = list()
    for dfname in dfnames:
        strain = DATASET_STRAIN_DICT[dfname]
        df = join_data(load_dataset(dfname))
        dfs.append(preprocess(strain, df, CUTOFF))
        if expected:
            expected_dfs.append(preprocess(strain, generate_expected_data(strain, df), 1))
    
    return dfs, expected_dfs
    

def join_data(df: pd.DataFrame)-> pd.DataFrame:
    '''
    
    '''
    return df.groupby(["Segment", "Start", "End"]).sum(["NGS_read_count"]).reset_index()


def load_mapped_reads(experiment: str):
    '''

    '''
    acc_nums = ACCNUMDICT[experiment]

    dfs = list()
    for acc_num, meta in acc_nums.items():
        path = os.path.join(DATAPATH, experiment, f"{acc_num}_mapped_reads_per_segment.csv")
        if not os.path.exists(path):
            path = os.path.join(DATAPATH, experiment, f"{acc_num}both_mapped_reads_per_segment.csv")
        df = pd.read_csv(path, dtype={"counts":"int64","segment": "string"}, na_values=["", "None"], keep_default_na=False)
        for m in meta.keys():
            df[m] = meta[m]
        dfs.append(df)
    concat_df = pd.concat(dfs)

    return concat_df


def load_all_mapped_reads(dfnames):
    '''
    
    '''
    mr_dfs = list()
    for experiment in dfnames:
        df = load_mapped_reads(experiment)
        mr_dfs.append(df)
    return mr_dfs


def get_sequence(strain: str, seg: str, full: bool=False)-> object:
    '''
        Loads a DNA sequence given the strain and segment.
        :param strain: name of the strain
        :param seg: name of the segment
        :param full: if True the whole Biopython Seq Object is returned
                     if False a string object is returned

        :return: Biopython Seq Object or str() of the sequence
    '''
    fasta_file = os.path.join(DATAPATH, "strain_segment_fastas", strain, f"{seg}.fasta")
    seq_obj = SeqIO.read(fasta_file, "fasta")
    if full:
        return seq_obj
    else:
        return str(seq_obj.seq.transcribe())

def get_seq_len(strain: str, seg: str)-> int:
    '''
        Calculates the length of a specific sequence given the strain and
        segment.
        :param strain: name of the strain
        :param seg: name of the segment

        :return: length of the sequence as int
    '''
    return len(get_sequence(strain, seg))

def get_p_value_symbol(p: float)-> str:
    '''
        Indicates the statistical significance by letters. Is used for plots.
        :param p: p-value of the test

        :return: letter indicating the significance level
    '''
    if p < 0.0001:
        return "****"
    elif p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return ""

def create_sequence_library(data_dict: dict)-> dict:
    '''
        Gets the raw loaded sequence data, which is a dict over all strains.
        In each dict the value is a data frame including DI RNA candidates.
        Creates the DI RNA sequence and adds it to the data frame.
        :param data_dict: dictionary key is strain names, value is df of DI RNA
                          candiates

        :return: dictionary with key for each strain. Value is a pandas df.
    '''
    for k, v in data_dict.items():
        del_seq_list = list()
        for i, row in v.iterrows():
            full_seq = get_sequence(k, row["Segment"])
            del_seq = full_seq[:row["Start"]] + full_seq[row["End"]-1:]
            del_seq_list.append(del_seq)

        data_dict[k]["DIRNASequence"] = del_seq_list

    return data_dict


#####################
### DIRECT REPATS ###
#####################
def calculate_direct_repeat(seq: str,
                            s: int,
                            e: int,
                            w_len: int
                            )-> Tuple[int, str]:
    '''
        Counts the number of overlapping nucleotides directly before start and
        end of junction site --> direct repeats
        :param seq: nucleotide sequence
        :param w_len: length of window to be searched
        :param s: start point
        :param e: end point

        :return: Tuple with two entries:
                    Integer giving the number of overlapping nucleotides
                    String of the overlapping nucleotides
    '''
    counter = 0

    start_window = seq[s-w_len: s]
    end_window = seq[e-1-w_len: e-1]
        
    #if they are the same return directly to avoid off-by-one error
    if start_window == end_window:
        return len(start_window), start_window

    if len(seq) < e:
        return 0, "_"
        
    for i in range(len(end_window) - 1, -1, -1):
        if start_window[i] == end_window[i]:
            counter += 1
        else:
            break
    overlap_seq = str(start_window[i+1:w_len])

    assert counter == len(overlap_seq), f"{counter=}, {len(overlap_seq)}"
    if len(overlap_seq) == 0:
        overlap_seq = "_"

    return counter, overlap_seq

def count_direct_repeats_overall(df: pd.DataFrame,
                                 seq: str
                                 )-> Tuple[dict, dict]:
    '''
        Calculates the number of direct repeats for each data point.
        :param df: dataframe with sequence and junction site data
        :param seq: RNA sequence of the given segement and strain
        :param mode: states which calculation mode is used in 
                     calculate_overlapping_nucleotides() check there for info

        :return: Tuple including a dict with the count of the length of
                 overlapping sequences and a dict with the overlapping
                 sequences and their count.
    '''
    w_len = 5
    nuc_overlap_dict = dict({i: 0 for i in range(0, w_len+1)})
    overlap_seq_dict = dict()
 
    for i, row in df.iterrows():
        s = row["Start"]
        e = row["End"]
        idx, overlap_seq = calculate_direct_repeat(seq, s, e, w_len)
        nuc_overlap_dict[idx] += 1
        if overlap_seq in overlap_seq_dict:
            overlap_seq_dict[overlap_seq] += 1
        else:
            overlap_seq_dict[overlap_seq] = 1

    return nuc_overlap_dict, overlap_seq_dict

#############################
### NUCLEOTIDE ENRICHMENT ###
#############################
def count_nucleotide_occurrence(seq: str,
                                p: int
                                )-> dict:
    '''
        Counts the number of nucleotides next to a given point.
        Goes 5 steps in both directions.
        :param seq: whole RNA sequence
        :param p: point on the sequence where to count

        :return: returns a counter dict with an entry for each nucleotide. In
                 each entry the counter for each position is given.
    '''
    window = seq[p-5:p+5]
    r_dict = dict({n: np.zeros(10) for n in NUCLEOTIDES})

    for i, char in enumerate(window):
        r_dict[char][i] = 1
    return r_dict

def count_nucleotide_occurrence_overall(df: pd.DataFrame,
                                        seq: str
                                        )-> Tuple[dict, dict]:
    '''
        Counts the occurrence of each nucleotide at different positions around
        the junction site
        :param df: dataframe with sequence and junction site data
        :param seq: rna sequence where to count the occurrence

        :return: tupel with two entries:
                    dict with nucleotide count for start site
                    dict with nucleotide count for end site
    '''

    count_start_dict = dict({n: np.zeros(10) for n in NUCLEOTIDES})
    count_end_dict = dict({n: np.zeros(10) for n in NUCLEOTIDES})
    normalize = 0

    for i, row in df.iterrows():
        seq_start_dict = count_nucleotide_occurrence(seq, row["Start"]) 
        seq_end_dict = count_nucleotide_occurrence(seq, row["End"]-1)
        normalize += 1
        for nuc in count_start_dict.keys():
            count_start_dict[nuc] += seq_start_dict[nuc]
            count_end_dict[nuc] += seq_end_dict[nuc]

    return count_start_dict, count_end_dict

#####################
### expected data ###
#####################
def generate_expected_data(strain: str, df: pd.DataFrame)-> pd.DataFrame:
    '''
    
    '''
    for seg in SEGMENTS:
        df = df.loc[df["Segment"] == seg]
        if len(df) == 0:
            continue
        seq = get_sequence(strain, seg)
        start = int(df["Start"].mean())
        end = int(df["End"].mean())
        s = (max(start-200, 50), start+200)
        e = (end-200, min(end+200, len(seq)-50))
        assert s[1] < e[0], "Sampling: start and end positions are overlapping!"
        # skip if there is no range given
        # this would lead to oversampling of a single position
        if s[0] == s[1] or e[0] == e[1]:
            continue
        if "samp_df" in locals():
            temp_df = generate_sampling_data(seq, s, e, N_SAMPLES)
            temp_df["Segment"] = seg
            samp_df = pd.concat([samp_df, temp_df], ignore_index=True)
        else:
            samp_df = generate_sampling_data(seq, s, e, N_SAMPLES)
            samp_df["Segment"] = seg
    
    samp_df["NGS_read_count"] = 1
    return samp_df

def generate_sampling_data(seq: str, s: Tuple[int, int], e: Tuple[int, int],  n: int)-> pd.DataFrame:
    '''
        Generates sampling data by creating random start and end points for
        artificial deletion sites. Generated data is used to calculate the
        expected values.
        :param seq: RNA sequence
        :param s: tuple with start and end point of the range for the artifical
                  start point of the deletion site
        :param e: tuple with start and end point of the range for the artifical
                  end point of the deletion site
        :param n: number of samples to generate

        :return: dataframe with the artifical data set
    '''
    # create all combinations of start and end positions that are possible
    combinations = [(x, y) for x in range(s[0], s[1] + 1) for y in range(e[0], e[1] + 1 )]

    # create for each the DI Sequence
    sequences = [seq[:start] + seq[end-1:] for (start, end) in combinations]

    # filter out duplicate DI sequences while keeping the ones with highest start number
    start, end = zip(*combinations)
    temp_df = pd.DataFrame(data=dict({"Start": start, "End": end, "Sequence": sequences}))

    # these are the direct repeat ratios that would be expected by chance overall
    # I commented them out because they are not used here, but still important to validate 
    #duplicates = temp_df.groupby('Sequence').size()
    #dir_rep_counts = duplicates.groupby(duplicates).size()
    #print(dir_rep_counts/sum(dir_rep_counts * dir_rep_counts.index))
    
    # Find the index of the row with the maximum value in the 'Start' column for each 'Sequence'
    max_start_index = temp_df.groupby('Sequence')['Start'].idxmax()
    # Use the index to select the corresponding rows from the original DataFrame
    result_df = temp_df.loc[max_start_index]
    # Replicate each row by the number of times it was found in the group
    result_df = result_df.loc[result_df.index.repeat(temp_df.groupby('Sequence').size())]
    df_no_duplicates = result_df.reset_index(drop=True).drop("Sequence", axis=1)

    # sample n of the remaining possible DI RNAs
    random_rows = df_no_duplicates.sample(n)
    return random_rows


#######################
### Data processing ###
#######################
def create_nucleotide_ratio_matrix(df, col):
    '''
    
    '''
    probability_matrix = pd.DataFrame(columns=NUCLEOTIDES.keys())
    seq_matrix = df.filter([col], axis=1)
    seq_matrix = seq_matrix[col].str.split("", expand=True)
    # drop first and last column
    seq_matrix = seq_matrix.drop([0, len(seq_matrix.columns)-1], axis=1)
    
    for n in NUCLEOTIDES.keys():
        probability_matrix[n] = seq_matrix.apply(lambda x: dict(x.value_counts()).get(n,0)/len(x), axis=0)

    return probability_matrix
def plot_heatmap(y,x,vals,ax, format=".2f", cmap="coolwarm", vmin=0, vmax=1, cbar=False,cbar_ax=None, cbar_kws=None):
    '''
        Plot heatmap for values in vals, with x (columns) and y (rows) as labels.
    '''
    df = pd.DataFrame({"x":x,"y":y,"vals":vals})
    df = pd.pivot_table(df, index="x", columns="y", values="vals", sort=False)
    ax = sns.heatmap(df, fmt=format, annot=True, vmin=vmin, vmax=vmax, ax=ax, cbar=cbar, cmap=cmap, cbar_ax=cbar_ax, cbar_kws=cbar_kws)
    return ax
def sequence_df(df, strain, isize=5):
    '''Generate a DataFrame with sequence information.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the DIP candidates in the "key" column. Nomenclature: {seg}_{start}_{end}
        isize (int, optional): The size of the sequence before and after the start and end positions. Default is 5.

    Returns:
        pandas.DataFrame: A DataFrame with the following columns:
            - "key": The original key from the input DataFrame.
            - "seg": The segment extracted from the key.
            - "start": The start position extracted from the key.
            - "end": The end position extracted from the key.
            - "seq": The dip sequence obtained from the "key".
            - "deleted_sequence": The deleted sequence obtained from the "key".
            - "isize": The specified size for the before and after sequences.
            - "seq_before_start": The sequence before the start position of length "isize".
            - "seq_after_start": The sequence after the start position of length "isize".
            - "seq_before_end": The sequence before the end position of length "isize".
            - "seq_after_end": The sequence after the end position of length "isize".
    '''
    res_df = pd.DataFrame(columns=["key","Segment", "Start","End","seq", "deleted_sequence", "isize", "full_seq", "Strain", "seq_around_deletion_junction", "NGS_read_count"])
    for _, r in df.iterrows():
        k = r["key"]
        seq, seq_head, seq_foot = get_dip_sequence(k, strain)
        start = int(k.split("_")[1].split("_")[0])
        end = int(k.split("_")[2])
        seg = k.split("_")[0]
        full_seq = get_sequence(strain, seg)
        deleted_seq = get_deleted_sequence(k, strain)
        seq_before_start = seq_head[-isize:]
        seq_after_start = deleted_seq[:isize]
        seq_before_end = deleted_seq[-isize:]
        seq_after_end = seq_foot[:isize]
        NGS_read_count = r["NGS_read_count"]

        seq_around_deletion_junction = seq_before_start + seq_after_start + seq_before_end + seq_after_end
        res_df = pd.concat([res_df, pd.DataFrame({"key":k, "Segment":seg, "Start":start, "End":end, "seq":seq, "isize":isize, "full_seq": full_seq, "Strain": strain,
                                "deleted_sequence":deleted_seq, "seq_around_deletion_junction": seq_around_deletion_junction, "NGS_read_count": NGS_read_count}, index=[0])], ignore_index=True)
    return res_df
def preprocess(strain, df, cutoff):
    '''
    
    '''
    if cutoff > 1:
        df = df[df["NGS_read_count"] >= cutoff].copy()
    df["key"] = df["Segment"] + "_" + df["Start"].map(str) + "_" + df["End"].map(str)
    return sequence_df(df, strain)
def get_deleted_sequence(dip_id, strain):
    '''
    return the sequence of a dip_id

    Args:
        dip_id (str): the id of the dip

    Returns:
        str: the sequence of the dip
    '''
    seg, start, end = dip_id.split("_")
    seq = get_sequence(strain, seg)
    return seq[int(start):int(end)-1]
def get_dip_sequence(dip_id, strain):
    '''
    
    '''
    seg, start, end = dip_id.split("_")
    fl_seq = get_sequence(strain, seg)
    seq_head = fl_seq[:int(start)]
    seq_foot = fl_seq[int(end)-1:]
    del_length = int(end)-int(start)
    return seq_head + "*"*del_length + seq_foot, seq_head, seq_foot