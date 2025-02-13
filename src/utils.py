'''
    General functions and global parameters, that are used in different scripts
'''
import os

import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats

from typing import Tuple
from Bio import SeqIO

### STATIC VALUES ###
# load config and assign values to global variables
DATAPATH = "/home/jens/DIPs/DIP_meta-study/data"
RESULTSPATH = "/home/jens/DIPs/DIP_meta-study/results"

# segments, nuclotides, and strains
CMAP = "Accent"
CUTOFF = 15
N_SAMPLES = 35000
RESULTSPATH = os.path.join(RESULTSPATH, f"cutoff_{CUTOFF}")
SEGMENTS = list(["PB2", "PB1", "PA", "HA", "NP", "NA", "M", "NS"])
NUCLEOTIDES = dict({"A": "Adenine", "C": "Cytosin", "G": "Guanine", "U": "Uracil"})


DATASET_STRAIN_DICT = dict({
    # H1N1
    "Alnaji2021": "PR8",
    "Pelz2021": "PR8",
    "Wang2023": "PR8",
    "Wang2020": "PR8",
    "Zhuravlev2020": "PR8",
    "Kupke2020": "PR8",
    "VdHoecke2015": "PR8",
    "Alnaji2019_Cal07": "Cal07",
    "Alnaji2019_NC" : "NC",
    "Mendes2021": "WSN_Mendes_rev",
    "Boussier2020": "WSN",
    # H3N2
    "Alnaji2019_Perth": "Perth",
    "Berry2021_A": "Connecticut",
    # H5N1
    "Penn2022": "Turkey",
    # H7N9
    "Lui2019": "Anhui",
    # B 
    "Alnaji2019_BLEE": "BLEE",
    "Berry2021_B": "Victoria",
    "Valesano2020_Vic": "Victoria",
    "Sheng2018": "Brisbane",
    "Berry2021_B_Yam": "Yamagata",
    "Southgate2019": "Yamagata",
    "Valesano2020_Yam": "Yamagata"
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
        "SRR8754523": dict({"Lineage": "2", "Passage": "6"}),
        "SRR8754531": dict({"Lineage": "1", "Passage": "6_t"}),
        "SRR8754532": dict({"Lineage": "1", "Passage": "3_t"}),
        "SRR8754533": dict({"Lineage": "1", "Passage": "1_t"})
    }),
    "Alnaji2019_NC": dict({
        "SRR8754513": dict({"Lineage": "2", "Passage": "1"}),
        "SRR8754514": dict({"Lineage": "1", "Passage": "1"}),
        "SRR8754527": dict({"Lineage": "1", "Passage": "6"}),
        "SRR8754538": dict({"Lineage": "2", "Passage": "6"})
    }),
    "Alnaji2019_Perth": dict({
        "SRR8754517": dict({"Lineage": "2", "Passage": "8"}),
        "SRR8754524": dict({"Lineage": "1", "Passage": "4"}),
        "SRR8754525": dict({"Lineage": "2", "Passage": "4"}),
        "SRR8754526": dict({"Lineage": "1", "Passage": "8"})
    }),
    "Alnaji2019_BLEE": dict({
        "SRR8754507": dict({"Lineage": "1", "Passage": "8"}),
        "SRR8754508": dict({"Lineage": "2", "Passage": "7"}),
        "SRR8754509": dict({"Lineage": "1", "Passage": "7"}),
        "SRR8754516": dict({"Lineage": "2", "Passage": "8"})
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
    "Zhuravlev2020": dict({
        "ERR4566024":  dict({"Cell": "A549"}),
        "ERR4566025":  dict({"Cell": "A549"}),
        "ERR4566028":  dict({"Cell": "HEK293FT"}),
        "ERR4566029":  dict({"Cell": "HEK293FT"}),
        "ERR4566032":  dict({"Cell": "MRC5"}),
        "ERR4566033":  dict({"Cell": "MRC5"}),
        "ERR4566036":  dict({"Cell": "WI38"}),
        "ERR4566037":  dict({"Cell": "WI38"})
    }),
    "Berry2021_A": dict({
        "SRR15182178":  dict({}),
        "SRR15182177":  dict({}),
        "SRR15182176":  dict({}),
        "SRR15182175":  dict({}),
        "SRR15182174":  dict({}),
        "SRR15182173":  dict({}),
        "SRR15182172":  dict({}),
        "SRR15182171":  dict({})
    }),
    "Berry2021_B": dict({
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
    "Berry2021_B_Yam": dict({
        "SRR15183338":  dict({}),
        "SRR15183343":  dict({}),
        "SRR15183342":  dict({}),
        "SRR15183341":  dict({}),
        "SRR15183340":  dict({}),
        "SRR15183339":  dict({})
    }),
    "Valesano2020_Vic": dict({
        "SRR10013092":  dict({}),
        "SRR10013237":  dict({}),
        "SRR10013181":  dict({}),
        "SRR10013242":  dict({}),
        "SRR10013050":  dict({}),
        "SRR10013272":  dict({}),
        "SRR10013047":  dict({}),
        "SRR10013239":  dict({}),
        "SRR10013071":  dict({}),
        "SRR10013201":  dict({}),
        "SRR10013072":  dict({}),
        "SRR10013200":  dict({}),
        "SRR10013108":  dict({}),
        "SRR10013256":  dict({}),
        "SRR10013037":  dict({}),
        "SRR10013254":  dict({}),
        "SRR10013279":  dict({}),
        "SRR10013219":  dict({}),
        "SRR10013221":  dict({})
    }),
    "Valesano2020_Yam": dict({
        "SRR10013243":  dict({}),
		"SRR10013084":  dict({}),
		"SRR10013188":  dict({}),
		"SRR10013094":  dict({}),
		"SRR10013178":  dict({}),
		"SRR10013236":  dict({}),
		"SRR10013063":  dict({}),
		"SRR10013209":  dict({}),
		"SRR10013241":  dict({}),
		"SRR10013240":  dict({}),
		"SRR10013229":  dict({}),
		"SRR10013068":  dict({}),
		"SRR10013205":  dict({}),
		"SRR10013067":  dict({}),
		"SRR10013206":  dict({}),
		"SRR10013062":  dict({}),
		"SRR10013210":  dict({}),
		"SRR10013070":  dict({}),
		"SRR10013203":  dict({}),
		"SRR10013103":  dict({}),
		"SRR10013170":  dict({}),
		"SRR10013223":  dict({}),
		"SRR10013244":  dict({}),
		"SRR10013275":  dict({})
    }),
    "Southgate2019": dict({
        "ERR3474616": dict({}),
        "ERR3474621": dict({}),
        "ERR3474642": dict({}),
        "ERR3474643": dict({}),
        "ERR3474658": dict({}),
        "ERR3474661": dict({}),
        "ERR3474662": dict({}),
        "ERR3474663": dict({}),
        "ERR3474664": dict({}),
        "ERR3474666": dict({}),
        "ERR3474671": dict({}),
        "ERR3474674": dict({}),
        "ERR3474675": dict({}),
        "ERR3474676": dict({}),
        "ERR3474679": dict({}),
        "ERR3474684": dict({}),
        "ERR3474685": dict({}),
        "ERR3474686": dict({}),
        "ERR3474687": dict({}),
        "ERR3474689": dict({}),
        "ERR3474692": dict({}),
        "ERR3474693": dict({}),
        "ERR3474694": dict({}),
        "ERR3474695": dict({}),
        "ERR3474697": dict({}),
        "ERR3474698": dict({}),
        "ERR3474699": dict({}),
        "ERR3474701": dict({}),
        "ERR3474702": dict({}),
        "ERR3474703": dict({}),
        "ERR3474704": dict({}),
        "ERR3474705": dict({}),
        "ERR3474706": dict({}),
        "ERR3474707": dict({}),
        "ERR3474709": dict({}),
        "ERR3474710": dict({}),
        "ERR3474712": dict({}),
        "ERR3474713": dict({}),
        "ERR3474714": dict({}),
        "ERR3474715": dict({}),
        "ERR3474716": dict({}),
        "ERR3474717": dict({}),
        "ERR3474718": dict({}),
        "ERR3474719": dict({}),
        "ERR3474720": dict({}),
        "ERR3474721": dict({}),
        "ERR3474722": dict({}),
        "ERR3474723": dict({}),
        "ERR3474724": dict({}),
        "ERR3474725": dict({}),
        "ERR3474726": dict({}),
        "ERR3474728": dict({}),
        "ERR3474729": dict({}),
        "ERR3474750": dict({}),
        "ERR3474751": dict({}),
        "ERR3474781": dict({}),
        "ERR3474796": dict({}),
        "ERR3474809": dict({})
    }),
    "VdHoecke2015": dict({
        "SRR1757953": dict({}),
        "SRR1758027": dict({})
    }),
    "Boussier2020": dict({
        "180628A_rec_A-P1p_S218": dict({}),
        "180628A_rec_B-P1p_S219": dict({}),
        "180628A_rec_C-P1p_S219": dict({}),
        "180628A_rec_D-P1p_S221": dict({}),
        "180628A_rec_WT1p6-1213_S242": dict({}),
        "180628B_rec_A-P1p-PCR_S213": dict({}),
        "180628B_rec_B-P1p-PCR_S214": dict({}),
        "180628B_rec_C-P1p-PCR_S215": dict({}),
        "180628B_rec_D-P1p-PCR_S216": dict({}),
        "180628B_rec_WT-P1p-PCR_S217": dict({}),
        "180705A_rec_AP1pb_S294": dict({}),
        "180705A_rec_BP1pb_S295": dict({}),
        "180705A_rec_CP1pb_S296": dict({}),
        "180705A_rec_DP1pb_S297": dict({}),
        "180705A_rec_WTP1pb_S298": dict({}),
        "180705B_rec_AP1pPCRb_S289": dict({}),
        "180705B_rec_BP1pPCRb_S290": dict({}),
        "180705B_rec_CP1pPCRb_S291": dict({}),
        "180705B_rec_DP1pPCRb_S292": dict({}),
        "180705B_rec_WTP1pPCRb_S293": dict({}),
        "180706A_rec_AP1pc_S10": dict({}),
        "180706A_rec_BP1pc_S11": dict({}),
        "180706A_rec_DP1pc_S12": dict({})
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
    "WSN_Mendes_rev": dict({
        "PB2_vRNA": "PB2",
        "PB1_vRNA": "PB1",
        "PA_vRNA": "PA",
        "HA_vRNA": "HA",
        "NP_vRNA": "NP",
        "NA_vRNA": "NA",
        "M_vRNA": "M",
        "NS_vRNA": "NS"
    }),
    "WSN": dict({
        "LC333182.1": "PB2",
        "LC333183.1": "PB1",
        "LC333184.1": "PA",
        "LC333185.1": "HA",
        "LC333186.1": "NP",
        "LC333187.1": "NA",
        "LC333188.1": "M",
        "LC333189.1": "NS"
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
    "H3N2_Thailand": dict({
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
        "OQ034430.1": "PB2",
        "OQ034429.1": "PB1",
        "OQ034431.1": "PA",
        "OQ034432.1": "HA",
        "OQ034433.1": "NP",
        "OQ034434.1": "NA",
        "OQ034435.1": "M",
        "OQ034436.1": "NS"
    }),
    "H1N1_Thailand":({
        "KU051428.1": "PB2",
        "KU051429.1": "PB1",
        "KU051430.1": "PA",
        "KU051431.1": "HA",
        "KU051432.1": "NP",
        "KU051433.1": "NA",
        "KU051434.1": "M",
        "KU051435.1": "NS"
    }),
    "Malaysia":({
        "CY040456.1": "PB2",
        "CY040455.1": "PB1",
        "CY040454.1": "PA",
        "CY040449.1": "HA",
        "CY040452.1": "NP",
        "CY040451.1": "NA",
        "CY040450.1": "M",
        "CY040453.1": "NS"
    })
})


### FUNCTIONS ###
def get_dataset_names(cutoff: int=0, selection: str="")-> list:
    '''
        Allows to select dataset names based on their cultivation type.
        :param cutoff: Threshold for min number of DelVGs in each dataset
        :param selection: cultivation type either 'in vivo mouse', 'in vitro'
                         or 'in vivo human'
        
        :return: list of dataset names
    '''
    if cutoff == 0 and selection == "":
        return list(DATASET_STRAIN_DICT.keys())
    
    path = os.path.join(RESULTSPATH, "metadata", f"dataset_stats_{CUTOFF}.csv")
    df = pd.read_csv(path)
    names = df[df["Size"] >= cutoff]["Dataset"].to_list()

    # make selection based on in vivo/cells etc.
    if selection == "in vivo mouse":
        select_names = ["Wang2023", "Penn2022", "Lui2019"]
    elif selection == "in vitro":
        select_names = ["Alnaji2021", "Pelz2021", "Wang2020", "Kupke2020", "Zhuravlev2020", "VdHoecke2015", "Alnaji2019_Cal07" ,"Alnaji2019_NC", "Mendes2021", "Boussier2020", "Alnaji2019_Perth", "Alnaji2019_BLEE", "Sheng2018"]
    elif selection == "in vivo human":
        select_names = ["Berry2021_A", "Berry2021_B", "Berry2021_B_Yam", "Southgate2019", "Valesano2020_Yam", "Valesano2020_Vic"]
    elif selection == "IAV":
        select_names = ["Alnaji2021", "Pelz2021", "Wang2023", "Wang2020", "Kupke2020", "Zhuravlev2020", "VdHoecke2015", "Alnaji2019_Cal07", "Alnaji2019_NC", "Mendes2021", "Boussier2020", "Alnaji2019_Perth", "Berry2021_A", "Penn2022", "Lui2019"]
    elif selection == "IBV":
        select_names = ["Alnaji2019_BLEE", "Berry2021_B", "Valesano2020_Vic", "Sheng2018", "Berry2021_B_Yam", "Southgate2019","Valesano2020_Yam"]
    else:
        select_names = names

    names = [name for name in names if name in select_names]
    return names

def load_single_dataset(exp: str, acc: str, segment_dict: dict)-> pd.DataFrame:
    '''
        Load a single dataset, defined by one SRA accession number.
        :param exp: name of the experiment (is also folder name)
        :param acc: SRA accession number
        :param segment_dict: dictionary that maps the ids of the reference
                            fastas to the segment names

        :return: Pandas Dataframe with one DelVG population
    '''
    path = os.path.join(DATAPATH, exp, f"{exp}_{acc}.csv")
    df = pd.read_csv(path,
                     dtype={"Segment": "string", "Start": "int64", "End": "int64", "NGS_read_count": "int64"},
                     na_values=["", "None"],
                     keep_default_na=False)
    df["Segment"] = df["Segment"].replace(segment_dict)

    return df

def load_dataset(dataset: str)-> pd.DataFrame:
    '''
        Load a full dataset, defined by multiple SRA accession numbers.
        :param exp: name of the experiment (is also folder name)

        :return: Pandas Dataframe with one DelVG population of whole experiment
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

def load_all(dfnames: list, expected: str=False)-> Tuple[list, list]:
    '''
        Load a list of datasets.
        :param dfnames: list of dataset names, each is one experiment
        :param expected: if True, expected data is loaded additionally

        :return: Tuple
            List of Pandas Dataframes each containing one experiment
            List of dataset names in same order as first list
    '''
    dfs = list()
    expected_dfs = list()
    for dfname in dfnames:
        strain = DATASET_STRAIN_DICT[dfname]
        df = join_data(load_dataset(dfname))
        dfs.append(preprocess(strain, df, CUTOFF))
        if expected:
            f = os.path.join(DATAPATH, "random_sampled", f"{dfname}_{CUTOFF}.csv")
            if os.path.exists(f):
                dtypes = {"Start": int, "End": int, "Segment": str, "NGS_read_count": int,
                          "key": str, "Strain": str, "isize": int, "full_seq": str,
                          "deleted_sequence": str, "seq_around_deletion_junction": str}
                exp_df = pd.read_csv(f, dtype=dtypes)
            else:
                df = df[df["NGS_read_count"] >= CUTOFF].copy()
                exp_df = preprocess(strain, generate_expected_data(strain, df), 1)
                exp_df.to_csv(f, index=False)
            expected_dfs.append(exp_df)
    return dfs, expected_dfs

def sort_datasets_by_type(dfs: list, dfnames: list, cutoff: int)-> Tuple[list, list]:
    '''
        Sorts a given name of experiments by cultivation type.
        :param dfs: list of datasets, ordered as in dfnames
        :param dfnames: list of dataset names, each is one experiment
        :param cutoff: Threshold for min number of DelVGs in each dataset

        :return: Tuple
            List of Pandas Dataframes each containing one experiment
            List of dataset names in same order as first list
    '''
    vitro = get_dataset_names(cutoff=cutoff, selection="in vitro")
    vivo = get_dataset_names(cutoff=cutoff, selection="in vivo mouse")
    patients = get_dataset_names(cutoff=cutoff, selection="in vivo human")
    dfnames_new_order = vitro + vivo + patients
    combined_data = list(zip(dfnames, dfs))

    def custom_sort(item):
        return dfnames_new_order.index(item[0])

    sorted_data = sorted(combined_data, key=custom_sort)
    dfnames_sorted, dfs_sorted = zip(*sorted_data)

    return dfs_sorted, dfnames_sorted

def join_data(df: pd.DataFrame)-> pd.DataFrame:
    '''
        Combine duplicate DelVGs and sum their NGS count.
        :param df: Pandas DataFrame with DelVG data

        :return: Pandas DataFrame without duplicate DelVGs
    '''
    return df.groupby(["Segment", "Start", "End"]).sum(["NGS_read_count"]).reset_index()

def load_mapped_reads(experiment: str)-> pd.DataFrame:
    '''
        Loads data about the reads that were mapped to each segment.
        :param experiment: name of the experiment (is also folder name)

        :return: Pandas DataFrame with mapped reads per segment
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

def load_all_mapped_reads(dfnames: list)-> list:
    '''
        Loads data about the mapped reads for all given experiments.
        :param dfnames: list of dataset names, each is one experiment

        :return: List of Pandas Dataframes each containing mapped reads for one
                experiment
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

##################
### STATISTICS ###
##################
def get_p_value_symbol(p: float)-> str:
    '''
        Indicates the statistical significance by strings. Is used for plots.
        :param p: p-value of the test

        :return: string indicating the significance level
    '''
    if p < 0.00001:
        return "***"
    elif p < 0.001:
        return "** "
    elif p < 0.05:
        return " * "
    else:
        return "ns."

def calc_cliffs_d(d1: list, d2: list)-> float:
    '''
        Cliff, Norman (1993). Dominance statistics: Ordinal analyses to answer
        ordinal questions (eq. 3)
        Cliffs d ranges from -1 (max effect of group 2) to 0 (no effect) to
        1 (max effect of group 1) Meissel K. and Yao E. (2024)
        :param d1: dataset 1
        :param d2: dataset 2

        :return: cliff's d
    '''
    U, p = stats.mannwhitneyu(d1, d2)
    print(f"U:\t{U}")
    print(f"\t{p}")
    cliffs_d = 2*U / (len(d1)*len(d2)) - 1
    return cliffs_d

def scheirer_ray_hare_test(data: pd.DataFrame)-> Tuple[float, float, float]:
    '''
        Calculates Scheirer Ray Hare test, which is the non-parametric
        alternative to the two-way ANOVA.
        :param data: Pandas dataframe with the data to be analyzed

        :return: Tuple
            H-statistic for factor IV type
            p-value for factor IV type
            H-statistic  for factor host system
            p-value for factor host system
            H-statistic for interaction of the two factors
            p-value for interaction of the two factors
    '''
    n_observations = len(data)
    data["rank"] = data["Measure"].rank()

    # calculating the sum of the squares
    rows = data.groupby(["IV_type"], as_index=False).agg({"rank":["count", "mean", "var"]}).rename(columns={"rank":"row"})
    rows.columns = ["_".join(col) for col in rows.columns]
    rows.columns = rows.columns.str.replace(r"_$","", regex=True)
    rows["sqdev"] = (rows.row_mean - rows.row_mean.mean())**2

    cols = data.groupby(["Host_system"], as_index=False).agg({"rank":["count", "mean", "var"]}).rename(columns={"rank":"col"})
    cols.columns = ["_".join(col) for col in cols.columns]
    cols.columns = cols.columns.str.replace(r"_$","", regex=True)
    cols["sqdev"] = (cols.col_mean-cols.col_mean.mean())**2

    data_sum         = data.groupby(["IV_type", "Host_system"], as_index=False).agg({"rank":["count", "mean", "var"]})
    data_sum.columns = ["_".join(col) for col in data_sum.columns]
    data_sum.columns = data_sum.columns.str.replace(r"_$","", regex=True)

    # Calculate sum of squares for each factor, interaction and mean of squares
    Rows_SS    = (rows["sqdev"] * rows["row_count"]).sum()
    Columns_SS = (cols["sqdev"] * cols["col_count"]).sum()
    Within_SS  = data_sum.rank_var.sum()*(data_sum.rank_count.min()-1)
    MS         = data["rank"].var()
    TOTAL_SS   = MS * (n_observations-1)
    Inter_SS   = TOTAL_SS - Within_SS - Rows_SS - Columns_SS

    # calculating the H-statistics and degrees of freedom
    H_rows = Rows_SS/MS
    H_cols = Columns_SS/MS
    H_int  = Inter_SS/MS
    df_rows   = len(rows)-1
    df_cols   = len(cols)-1
    df_int    = df_rows*df_cols
    
    # calculating the p-values
    p_rows  = 1-stats.chi2.cdf(H_rows, df_rows)
    p_cols  = 1-stats.chi2.cdf(H_cols, df_cols)
    p_inter = 1-stats.chi2.cdf(H_int, df_int)

    return H_rows, p_rows, H_cols, p_cols, H_int, p_inter

def get_eta_squared(H, k, n):
    '''
        Based on Cohen (2008), Explaining psychological statistics
    '''
    eta = (H - k + 1)/(n - k)
    return eta


######################
### DIRECT REPEATS ###
######################
def calculate_direct_repeat(seq: str, s: int, e: int, w_len: int)-> Tuple[int, str]:
    '''
        Counts the number of overlapping nucleotides directly before start and
        end of junction site --> direct repeats
        :param seq: nucleotide sequence
        :param s: start point
        :param e: end point
        :param w_len: length of window to be searched

        :return: Tuple
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
        
    for i in range(len(end_window)-1, -1, -1):
        if start_window[i] == end_window[i]:
            counter += 1
        else:
            break
    overlap_seq = str(start_window[i+1:w_len])

    assert counter == len(overlap_seq), f"{counter=}, {len(overlap_seq)}"
    if len(overlap_seq) == 0:
        overlap_seq = "_"

    return counter, overlap_seq

def count_direct_repeats_overall(df: pd.DataFrame, seq: str)-> Tuple[dict, dict]:
    '''
        Calculates the number of direct repeats for each data point.
        :param df: dataframe with sequence and junction site data
        :param seq: RNA sequence of the given segement and strain

        :return: Tuple
            Dict with the count of the direct repeat lengths
            Dict with the overlapping sequences and their count
    '''
    w_len = 5
    nuc_overlap_dict = dict({i: 0 for i in range(0, w_len+1)})
    overlap_seq_dict = dict()
 
    for _, row in df.iterrows():
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
def count_nucleotide_occurrence(seq: str, p: int)-> dict:
    '''
        Counts the number of nucleotides next to a given point.
        Goes 5 steps in both directions.
        :param seq: whole RNA sequence
        :param p: point on the sequence where to count

        :return: Counter dict with an entry for each nucleotide. In each entry
                the counter for each position is given.
    '''
    window = seq[p-5:p+5]
    r_dict = dict({n: np.zeros(10) for n in NUCLEOTIDES})

    for i, char in enumerate(window):
        r_dict[char][i] = 1
    return r_dict

def count_nucleotide_occurrence_overall(df: pd.DataFrame, seq: str)-> Tuple[dict, dict]:
    '''
        Counts the occurrence of each nucleotide at different positions around
        the junction site
        :param df: dataframe with sequence and junction site data
        :param seq: rna sequence where to count the occurrence

        :return: Tuple
            Dict with nucleotide count for start of deletion site
            Dict with nucleotide count for end of deletion site
    '''

    count_start_dict = dict({n: np.zeros(10) for n in NUCLEOTIDES})
    count_end_dict = dict({n: np.zeros(10) for n in NUCLEOTIDES})
    normalize = 0

    for _, row in df.iterrows():
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
        Randomly samples deletion sites for a given dataset which can be used
        to compare the results of the real dataset.
        :param strain: name of the strain
        :param df: DelVG dataset

        :return: artifical dataset that includes random deletion sites
    '''
    for seg in SEGMENTS:
        df_s = df.loc[df["Segment"] == seg]
        if len(df_s) == 0:
            continue
        seq = get_sequence(strain, seg)
        start = int(df_s["Start"].mean())
        end = int(df_s["End"].mean())
        s = (max(start-200, 50), start+200)
        e = (end-200, min(end+200, len(seq)-50))
        
        # skip if there is no range given this would lead to oversampling of a single position
        if s[0] == s[1] or e[0] == e[1]:
            continue
        # positions are overlapping
        if s[1] > e[0]:
            continue
        if "samp_df" in locals():
            temp_df = generate_sampling_data(seq, s, e, N_SAMPLES)
            temp_df["Segment"] = seg
            samp_df = pd.concat([samp_df, temp_df], ignore_index=True)
        else:
            samp_df = generate_sampling_data(seq, s, e, N_SAMPLES)
            samp_df["Segment"] = seg
    
    samp_df["NGS_read_count"] = 1
    return samp_df.reset_index()

def generate_sampling_data(seq: str, s: Tuple[int, int], e: Tuple[int, int],  n: int)-> pd.DataFrame:
    '''
        Generates sampling data by creating random start and end points for
        artificial deletion sites. Generated data is used to calculate the
        expected values.
        :param seq: RNA sequence
        :param s: tuple with start and end point of the range for the artifical
                  start point of the deletion sites
        :param e: tuple with start and end point of the range for the artifical
                  end point of the deletion sites
        :param n: number of samples to generate

        :return: Pandas DataFrame of the artifical data set
    '''
    df_no_duplicates = create_sampling_space(seq, s, e)
    return df_no_duplicates.sample(n)

def create_sampling_space(seq: str, s: Tuple[int, int], e: Tuple[int, int])-> pd.DataFrame:
    '''
        Creates all possible candidates that would be expected.
        :param seq: RNA sequence
        :param s: tuple with start and end point of the range for the artifical
                  start point of the deletion sites
        :param e: tuple with start and end point of the range for the artifical
                  end point of the deletion sites
        
        :return: dataframe with possible DelVG candidates
    '''
    # create all combinations of start and end positions that are possible
    combinations = [(x, y) for x in range(s[0], s[1]+1) for y in range(e[0], e[1]+1)]

    # create for each the DelVG Sequence
    sequences = [seq[:start] + seq[end-1:] for (start, end) in combinations]

    # filter out duplicate DelVG sequences while keeping the ones with highest start number
    start, end = zip(*combinations)
    temp_df = pd.DataFrame(data=dict({"Start": start, "End": end, "Sequence": sequences}))

    # Find the index of the row with the maximum value in the "Start" column for each "Sequence"
    max_start_index = temp_df.groupby("Sequence")["Start"].idxmax()
    result_df = temp_df.loc[max_start_index]
    # Replicate each row by the number of times it was found in the group
    result_df = result_df.loc[result_df.index.repeat(temp_df.groupby("Sequence").size())]
    df_no_duplicates = result_df.reset_index(drop=True).drop("Sequence", axis=1)

    return df_no_duplicates

#######################
### Data processing ###
#######################
def create_nucleotide_ratio_matrix(df: pd.DataFrame, col: str)-> pd.DataFrame:
    '''
        Counts nucleotides around the deletion site. Used to create heatmaps.
        :param df: Pandas DataFrame that was created using sequence_df()
        :param col: column name which sequence to use

        :return: Pandas DataFrame with probabilites for the nucleotides
    '''
    probability_matrix = pd.DataFrame(columns=NUCLEOTIDES.keys())
    seq_matrix = df.filter([col], axis=1)
    seq_matrix = seq_matrix[col].str.split("", expand=True)
    # drop first and last column
    seq_matrix = seq_matrix.drop([0, len(seq_matrix.columns)-1], axis=1)
    
    for n in NUCLEOTIDES.keys():
        probability_matrix[n] = seq_matrix.apply(lambda x: dict(x.value_counts()).get(n,0)/len(x), axis=0)

    return probability_matrix

def plot_heatmap(y: list, x: list, vals: list, ax: object,
                 format=".2f", cmap="coolwarm", vmin=0, vmax=1, cbar=False, cbar_ax=None, cbar_kws=None)-> object:
    '''
        Helper function to plot heatmap.
        :param y: columns of heatmap
        :param x: rows of heatmap
        :param vals: values for heatmap
        :param ax: matplotlib.axes object
        :param: additional parameters check sns.heatmap() for more information
        
        :return: generated heatmap on matplotlib.axes object
    '''
    df = pd.DataFrame({"x":x,"y":y,"vals":vals})
    df = pd.pivot_table(df, index="x", columns="y", values="vals", sort=False)
    ax = sns.heatmap(df, fmt=format, annot=True, vmin=vmin, vmax=vmax, ax=ax, cbar=cbar, cmap=cmap, cbar_ax=cbar_ax, cbar_kws=cbar_kws)
    return ax

def sequence_df(df: pd.DataFrame, strain: str, isize: int=5)-> pd.DataFrame:
    '''
        Generate a DataFrame with sequence information.
        :param df: Pandas DataFrame containing the DelVGs in the "key" column
            Nomenclature: {seg}_{start}_{end}
        :param strain: name of the strain
        :param isize: the size of the sequence before and after the start and
            end positions. Default is 5.

    :return: Pandas DataFrame with the following columns:
            - "key": The original key from the input DataFrame.
            - "Segment": The segment
            - "Start": The start position of the deletion site
            - "End": The end position of the deletion site
            - "seq": The dip sequence
            - "deleted_sequence": The deleted sequence
            - "isize": The specified size for the before and after sequences
            - "full_seq": full sequence of the wild type virus
            - "Strain": strain used in the experiment
            - "seq_around_deletion_junction": sequence around deletion sites
            - "NGS_read_count": NGS count measured in the experiment

    '''
    df["Strain"] = strain
    df["Start"] = df.apply(lambda row: int(row["key"].split("_")[1]), axis=1)
    df["End"] = df.apply(lambda row: int(row["key"].split("_")[2]), axis=1)
    df["Segment"] = df.apply(lambda row: row["key"].split("_")[0], axis=1)
    df["isize"] = isize
    def wrap_get_sequence(row):
        return get_sequence(row["Strain"], row["Segment"])
    df["full_seq"] = df.apply(wrap_get_sequence, axis=1)
    def wrap_get_deleted_sequence(row):
        return get_deleted_sequence(row["key"], row["Strain"])
    df["deleted_sequence"] = df.apply(wrap_get_deleted_sequence, axis=1)
    def get_seq_around_del(row):
        seq_head = get_dip_sequence(row["key"], row["Strain"])[1]
        seq_foot = get_dip_sequence(row["key"], row["Strain"])[2]
        
        seq_before_start = seq_head[-row["isize"]:]
        seq_after_start = row["deleted_sequence"][:row["isize"]]
        seq_before_end = row["deleted_sequence"][-row["isize"]:]
        seq_after_end = seq_foot[:row["isize"]]
        return seq_before_start + seq_after_start + seq_before_end + seq_after_end
    df["seq_around_deletion_junction"] = df.apply(get_seq_around_del, axis=1)
    return df

def preprocess(strain: str, df: pd.DataFrame, thresh: int)-> pd.DataFrame:
    '''
        Excluding DelVGs with to low NGS count and running sequence_df().
        :param strain: name of the strain
        :param df: Pandas DataFrame with DelVG data
        :param thresh: Threshold for min number of count for each DelVG

        :return: resulting df of sequence_df() function
    '''
    if thresh > 1:
        df = df[df["NGS_read_count"] >= thresh].copy()
    df["key"] = df["Segment"] + "_" + df["Start"].map(str) + "_" + df["End"].map(str)
    return sequence_df(df, strain)

def get_deleted_sequence(delvg_id: str, strain: str)-> str:
    '''
        Return the sequence of the deletion site.
        :param delvg_id: the id of the DelVG ({seg}_{start}_{end})
        :param strain: name of the strain

        :return: the sequence that is deleted in a DelVG
    '''
    seg, start, end = delvg_id.split("_")
    seq = get_sequence(strain, seg)
    return seq[int(start):int(end)-1]

def get_dip_sequence(delvg_id: str, strain: str)-> Tuple[str, str, str]:
    '''
        Return the remaining sequence of a DelVG. Deletion is filled with "*".
        :param delvg_id: the id of the DelVG ({seg}_{start}_{end})
        :param strain: name of the strain

        :return: Tuple
            the remaining sequence of a DelVG
            the sequence before the deletion site
            the sequence after the deletion site
    '''
    seg, start, end = delvg_id.split("_")
    fl_seq = get_sequence(strain, seg)
    seq_head = fl_seq[:int(start)]
    seq_foot = fl_seq[int(end)-1:]
    del_length = int(end)-int(start)
    return seq_head + "*"*del_length + seq_foot, seq_head, seq_foot