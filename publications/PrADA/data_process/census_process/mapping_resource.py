workclass_map = {"Not in universe": "None",
                 "Private": "Private",
                 "Self-employed-not incorporated": "Self-emp-not-inc",
                 "Local government": "Local-gov",
                 "State government": "State-gov",
                 "Self-employed-incorporated": "Self-emp-inc",
                 "Federal government": "Federal-gov",
                 "Never worked": "Never-worked",
                 "Without pay": "Without-pay"}

education_map = {
    "Children": "Children",
    "Less than 1st grade": "Preschool",
    "1st 2nd 3rd or 4th grade": "1st-4th",
    "5th or 6th grade": "5th-6th",
    "7th and 8th grade": "7th-8th",
    "9th grade": "9th",
    "10th grade": "10th",
    "11th grade": "11th",
    "12th grade no diploma": "12th",
    "High school graduate": "HS-grad",
    "Some college but no degree": "Some-college",
    "Associates degree-academic program": "Assoc-acdm",
    "Associates degree-occup /vocational": "Assoc-voc",
    "Bachelors degree(BA AB BS)": "Bachelors",
    "Masters degree(MA MS MEng MEd MSW MBA)": "Masters",
    "Prof school degree (MD DDS DVM LLB JD)": "Prof-school",
    "Doctorate degree(PhD EdD)": "Doctorate"
}

race_map = {
    'White': "White",
    'Asian or Pacific Islander': "Asian-Pac-Islander",
    'Amer Indian Aleut or Eskimo': "Amer-Indian-Eskimo",
    'Black': "Black",
    'Other': "Other"}

marital_status_map = {"Widowed": "Widowed",
                      "Divorced": "Divorced",
                      "Never married": "Never-married",
                      "Married-civilian spouse present": "Married-civ-spouse",
                      "Separated": "Separated",
                      "Married-spouse absent": "Married-spouse-absent",
                      "Married-A F spouse present": "Married-AF-spouse"}

occupation_map = {
    'Not in universe': "Other-service",
    'Precision production craft & repair': "Craft-repair",
    'Professional specialty': "Prof-specialty",
    'Executive admin and managerial': "Exec-managerial",
    'Handlers equip cleaners etc': "Handlers-cleaners",
    'Handlers equip cleaners etc ': "Handlers-cleaners",
    'Adm support including clerical': "Adm-clerical",
    'Machine operators assmblrs & inspctrs': "Machine-op-inspct",
    'Other service': "Other-service",
    'Sales': "Sales",
    'Private household services': "Priv-house-serv",
    'Technicians and related support': "Tech-support",
    'Transportation and material moving': "Transport-moving",
    'Farming forestry and fishing': "Farming-fishing",
    'Protective services': "Protective-serv",
    'Armed Forces': "Armed-Forces"}

country_map = {
    'None': "None",
    'United-States': "United-States",
    'Vietnam': "Vietnam", 'Columbia': "Columbia", 'Mexico': "Mexico", 'Peru': "Peru",
    'Cuba': "Cuba", 'Philippines': "Philippines", 'Dominican-Republic': "Dominican-Republic",
    'El-Salvador': "El-Salvador", 'Canada': "Canada", 'Scotland': "Scotland", 'Portugal': "Portugal",
    'Guatemala': "Guatemala", 'Ecuador': "Ecuador", 'Germany': "Germany",
    'Outlying-U S (Guam USVI etc)': "Outlying-US(Guam-USVI-etc)",
    'Puerto-Rico': "Puerto-Rico", 'Italy': "Italy", 'China': "China", 'Poland': "Poland", 'Nicaragua': "Nicaragua",
    'Taiwan': "Taiwan", 'England': "England", 'Ireland': "Ireland", 'South Korea': "South-Korea",
    'Trinadad&Tobago': "Trinadad&Tobago",
    'Jamaica': "Jamaica", 'Honduras': "Honduras", 'Iran': "Iran", 'Hungary': "Hungary", 'France': "France",
    'Cambodia': "Cambodia",
    'India': "India", 'Hong Kong': "Hong-Kong", 'Japan': "Japan", 'Haiti': "Haiti",
    'Holand-Netherlands': "Holand-Netherlands", 'Greece': "Greece",
    'Thailand': "Thailand", 'Panama': "Panama", 'Yugoslavia': "Yugoslavia", 'Laos': "Laos"}

census_income_label = {"- 50000.": '<=50K', "50000+.": '>50K'}

workclass_index_map = {"None": 0,
                       "Private": 1,
                       "Self-emp-not-inc": 2,
                       "Local-gov": 3,
                       "State-gov": 4,
                       "Self-emp-inc": 5,
                       "Federal-gov": 6,
                       "Never-worked": 7,
                       "Without-pay": 8}

marital_status_index_map = {
    "Married-civ-spouse": 0,
    "Married-spouse-absent": 0,
    "Married-AF-spouse": 0,
    "Widowed": 1,
    "Divorced": 2,
    "Separated": 3,
    "Never-married": 4}

relationship_index_map = {
    'Other-relative': 0,
    'Not-in-family': 1,
    'Husband': 2,
    'Wife': 3,
    'Own-child': 4,
    'Unmarried': 5
}

education_index_map = {
    "Children": 0,
    "Preschool": 1,
    "1st-4th": 2,
    "5th-6th": 3,
    "7th-8th": 3,
    "9th": 4,
    "10th": 4,
    "11th": 4,
    "12th": 4,
    "HS-grad": 5,
    "Some-college": 6,
    "Assoc-acdm": 7,
    "Assoc-voc": 7,
    "Bachelors": 8,
    "Masters": 9,
    "Prof-school": 10,
    "Doctorate": 11,
}

# country_index_map = {
#     "None": 0,
#     "United-States": 1, "Vietnam": 2, "Columbia": 3, "Mexico": 4, "Peru": 5,
#     "Cuba": 6, "Philippines": 7, "Dominican-Republic": 8,
#     "El-Salvador": 9, "Canada": 10, "Scotland": 11, "Portugal": 12,
#     "Guatemala": 13, "Ecuador": 14, "Germany": 15,
#     "Outlying-US(Guam-USVI-etc)": 16, "Puerto-Rico": 17, "Italy": 18, "China": 19, "Poland": 20,
#     "Nicaragua": 21, "Taiwan": 22, "England": 23, "Ireland": 24, "South-Korea": 25, "Trinadad&Tobago": 26,
#     "Jamaica": 27, "Honduras": 28, "Iran": 29, "Hungary": 30, "France": 31, "Cambodia": 32,
#     "India": 33, "Hong-Kong": 34, "Japan": 35, "Haiti": 36, "Holand-Netherlands": 37, "Greece": 38,
#     "Thailand": 39, "Panama": 40, "Yugoslavia": 41, "Laos": 42}


country_index_map = {
    'None': 0,
    'United-States': 1,
    'Cambodia': 5,
    'Canada': 1,
    'China': 4,
    'Columbia': 3,
    'Cuba': 3,
    'Dominican-Republic': 3,
    'Ecuador': 3,
    'El-Salvador': 3,
    'England': 2,
    'France': 2,
    'Germany': 2,
    'Greece': 2,
    'Guatemala': 3,
    'Haiti': 3,
    'Holand-Netherlands': 2,
    'Honduras': 3,
    'Hong-Kong': 4,
    'Hungary': 2,
    'India': 5,
    'Iran': 5,
    'Ireland': 2,
    'Italy': 2,
    'Jamaica': 3,
    'Japan': 4,
    'Laos': 5,
    'Mexico': 3,
    'Nicaragua': 3,
    'Outlying-US(Guam-USVI-etc)': 1,
    'Panama': 3,
    'Peru': 3,
    'Philippines': 5,
    'Poland': 2,
    'Portugal': 2,
    'Puerto-Rico': 3,
    'Scotland': 2,
    'South-Korea': 4,
    'Taiwan': 4,
    'Thailand': 5,
    'Trinadad&Tobago': 3,
    'Vietnam': 4,
    'Yugoslavia': 2}

state_index_map = {
    "None": 0,
    "Not in universe": 0,
    "Maine": 1,
    "New Hampshire": 1,
    "Vermont": 1,
    "Massachusetts": 1,
    "Rhode Island": 1,
    "Connecticut": 2,
    "New York": 3,
    "New Jersey": 4,
    "Pennsylvania": 5,
    "Ohio": 6,
    "Indiana": 7,
    "Illinois": 8,
    "Michigan": 9,
    "Wisconsin": 9,
    "Minnesota": 10,
    "Iowa": 10,
    "Missouri": 10,
    "North Dakota": 10,
    "South Dakota": 10,
    "Nebraska": 10,
    "Kansas": 10,
    "Delaware": 11,
    "Virginia": 11,
    "Maryland": 11,
    "West Virginia": 11,
    "Washington,D.C.": 12,
    "North Carolina": 13,
    "South Carolina": 13,
    "Georgia": 13,
    "Florida": 14,
    "Kentucky": 15,
    "Tennessee": 15,
    "Alabama": 16,
    "Mississippi": 16,
    "Arkansas": 17,
    "Oklahoma": 17,
    "Louisiana": 17,
    "Texas": 18,
    "Montana": 19,
    "Idaho": 19,
    "Wyoming": 19,
    "Utah": 19,
    "Nevada": 19,
    "Colorado": 19,
    "New Mexico": 19,
    "Arizona": 19,
    "Washington": 20,
    "Alaska": 20,
    "Hawaii": 20,
    "Oregon": 20,
    "California": 21,
    "District of Columbia": 22,
    "Abroad": 23}

industry_index_map = {
    'Not in universe or children': 0,
    'Retail trade': 1,
    'Manufacturing-durable goods': 2,
    'Education': 3,
    'Manufacturing-nondurable goods': 4,
    'Finance insurance and real estate': 5,
    'Construction': 6,
    'Business and repair services': 7,
    'Medical except hospital': 8,
    'Public administration': 9,
    'Other professional services': 10,
    'Transportation': 11,
    'Hospital services': 12,
    'Wholesale trade': 13,
    'Agriculture': 14,
    'Personal services except private HH': 15,
    'Social services': 16,
    'Entertainment': 17,
    'Communications': 18,
    'Utilities and sanitary services': 19,
    'Private household services': 20,
    'Mining': 21,
    'Forestry and fisheries': 22,
    'Armed Forces': 23
}

occupation_index_map = {
    "None": 0,
    "Other-service": 1,
    "Craft-repair": 2,
    "Prof-specialty": 3,
    "Exec-managerial": 4,
    "Handlers-cleaners": 5,
    "Adm-clerical": 6,
    "Machine-op-inspct": 7,
    "Sales": 8,
    "Priv-house-serv": 9,
    "Tech-support": 10,
    "Transport-moving": 11,
    "Farming-fishing": 12,
    "Protective-serv": 13,
    "Armed-Forces": 14}

race_index_map = {
    "Other": 0,
    "White": 1,
    "Asian-Pac-Islander": 2,
    "Amer-Indian-Eskimo": 3,
    "Black": 4}

hisp_origin_index_map = {
    'None': 0,
    'Do not know': 0,
    'All other': 1,
    'Mexican-American': 2,
    'Mexican (Mexicano)': 3,
    'Central or South American': 4,
    'Puerto Rican': 5,
    'Other Spanish': 6,
    'Cuban': 7,
    'Chicano': 8
}

gender_index_map = {
    'Female': 0,
    'Male': 1
}

union_member_index_map = {
    'Not in universe': 0,
    'No': 1,
    'Yes': 2
}

unemp_reason_index_map = {
    'Not in universe': 0,
    'Other job loser': 1,
    'Re-entrant': 2,
    'Job loser - on layoff': 3,
    'Job leaver': 4,
    'New entrant': 5
}

full_or_part_emp_index_map = {
    'Children or Armed Forces': 0,
    'Full-time schedules': 1,
    'Not in labor force': 2,
    'PT for non-econ reasons usually FT': 3,
    'Unemployed full-time': 4,
    'PT for econ reasons usually PT': 5,
    'Unemployed part- time': 6,
    'PT for econ reasons usually FT': 7
}

tax_filer_stat_index_map = {
    'Nonfiler': 0,
    'Joint both under 65': 1,
    'Single': 2,
    'Joint both 65+': 3,
    'Head of household': 4,
    'Joint one under 65 & one 65+': 5
}

region_prev_res_index_map = {
    'Not in universe': 0,
    'South': 1,
    'West': 2,
    'Midwest': 3,
    'Northeast': 4,
    'Abroad': 5
}

# det_hh_fam_stat and det_hh_summ
det_hh_index_map = {
    'Householder': 0,
    'Child <18 never marr not in subfamily': 1,
    'Spouse of householder': 2,
    'Nonfamily householder': 3,
    'Child 18+ never marr Not in a subfamily': 4,
    'Secondary individual': 5,
    'Other Rel 18+ ever marr not in subfamily': 6,
    'Grandchild <18 never marr child of subfamily RP': 7,
    'Other Rel 18+ never marr not in subfamily': 8,
    'Grandchild <18 never marr not in subfamily': 9,
    'Child 18+ ever marr Not in a subfamily': 10,
    'Child under 18 of RP of unrel subfamily': 11,
    'RP of unrelated subfamily': 12,
    'Child 18+ ever marr RP of subfamily': 13,
    'Other Rel 18+ ever marr RP of subfamily': 14,
    'Other Rel <18 never marr child of subfamily RP': 15,
    'Other Rel 18+ spouse of subfamily RP': 16,
    'Child 18+ never marr RP of subfamily': 17,
    'Other Rel <18 never marr not in subfamily': 18,
    'Grandchild 18+ never marr not in subfamily': 19,
    'In group quarters': 20,
    'Child 18+ spouse of subfamily RP': 21,
    'Other Rel 18+ never marr RP of subfamily': 22,
    'Child <18 never marr RP of subfamily': 23,
    'Spouse of RP of unrelated subfamily': 24,
    'Child <18 ever marr not in subfamily': 25,
    'Grandchild 18+ ever marr not in subfamily': 26,
    'Grandchild 18+ spouse of subfamily RP': 27,
    'Child <18 ever marr RP of subfamily': 28,
    'Grandchild 18+ ever marr RP of subfamily': 29,
    'Grandchild 18+ never marr RP of subfamily': 30,
    'Other Rel <18 ever marr RP of subfamily': 31,
    'Other Rel <18 never married RP of subfamily': 32,
    'Other Rel <18 spouse of subfamily RP': 33,
    'Child <18 spouse of subfamily RP': 34,
    'Grandchild <18 ever marr not in subfamily': 35,
    'Grandchild <18 never marr RP of subfamily': 36,
    'Other Rel <18 ever marr not in subfamily': 37,
    'Child under 18 never married': 38,
    'Child 18 or older': 39,
    'Other relative of householder': 40,
    'Nonrelative of householder': 41,
    'Group Quarters- Secondary individual': 42,
    'Child under 18 ever married': 43
}

mig_chg_index_map = {
    'None': 0,
    'Nonmover': 1,
    'Not in universe': 2,
    'Not identifiable': 2,
    'MSA to MSA': 3,
    'NonMSA to nonMSA': 4,
    'MSA to nonMSA': 5,
    'NonMSA to MSA': 6,
    'Abroad to MSA': 7,
    'Abroad to nonMSA': 8,
    'Same county': 9,
    'Different county same state': 10,
    'Different region': 11,
    'Different state same division': 12,
    'Different division same region': 13,
    'Different state in South': 14,
    'Different state in West': 15,
    'Different state in Midwest': 16,
    'Different state in Northeast': 17,
    'Abroad': 18
}

mig_same_index_map = {
    'None': 0,
    'Not in universe under 1 year old': 1,
    'Yes': 2,
    'No': 3
}

mig_prev_sunbelt_index_map = {
    'None': 0,
    'Not in universe': 1,
    'No': 2,
    'Yes': 3
}

fam_under_18_index_map = {
    'Not in universe': 0,
    'Both parents present': 1,
    'Mother only present': 2,
    'Father only present': 3,
    'Neither parent present': 4
}

citizenship_index_map = {
    'Native- Born in the United States': 0,
    'Foreign born- Not a citizen of U S': 1,
    'Foreign born- Not a citizen of U S ': 1,
    'Foreign born- U S citizen by naturalization': 2,
    'Native- Born abroad of American Parent(s)': 3,
    'Native- Born in Puerto Rico or U S Outlying': 4
}

vet_question_index_map = {
    'Not in universe': 0,
    'No': 1,
    'Yes': 2
}

income_label_index = {'<=50K': 0, '>50K': 1}

test_income_label_index = {'<=50K.': 0, '>50K.': 1}

education_value_map = {
    "Children": 0,
    "Preschool": 0.5,
    "1st-4th": 2.5,
    "5th-6th": 5.5,
    "7th-8th": 7.5,
    "9th": 9,
    "10th": 10,
    "11th": 11,
    "12th": 12,
    "HS-grad": 12,
    "Some-college": 14,
    "Assoc-acdm": 14,
    "Assoc-voc": 14,
    "Bachelors": 16,
    "Masters": 18,
    "Prof-school": 20,
    "Doctorate": 21,
}

continuous_cols = ["age", "gender", "education_year", "capital_gain", "capital_loss"]
categorical_cols = ["class_worker", "major_ind_code", "major_occ_code", "unemp_reason", "full_or_part_emp",
                    "own_or_self",
                    "education", "race", "age_index", "gender_index", "marital_stat", "union_member", "vet_benefits",
                    "vet_question",
                    "region_prev_res", "state_prev_res", "mig_chg_msa", "mig_chg_reg", "mig_move_reg", "mig_same",
                    "mig_prev_sunbelt",
                    "tax_filer_stat", "det_hh_fam_stat", "det_hh_summ", "fam_under_18",
                    "hisp_origin", "country_father", "country_mother", "country_self", "citizenship"]
target_col_name = "income_label"

feature_group_map = {"employment": {"class_worker", "major_ind_code", "major_occ_code", "unemp_reason",
                                    "full_or_part_emp", "own_or_self"},
                     "demo": {"education", "race", "age_index", "gender_index", "marital_stat",
                              "union_member", "vet_benefits", "vet_question"},
                     "residence": {"region_prev_res", "state_prev_res", "mig_chg_msa",
                                   "mig_chg_reg", "mig_move_reg",
                                   "mig_same", "mig_prev_sunbelt"},
                     "household": {"tax_filer_stat", "det_hh_fam_stat", "det_hh_summ", "fam_under_18"},
                     "Origin": {"hisp_origin", "country_father", "country_mother", "country_self", "citizenship"}}

cate_to_index_map = {
    "class_worker": workclass_index_map,
    "education": education_index_map,
    "marital_stat": marital_status_index_map,
    "major_ind_code": industry_index_map,
    "major_occ_code": occupation_index_map,
    "race": race_index_map,
    "hisp_origin": hisp_origin_index_map,
    "gender": gender_index_map,
    "union_member": union_member_index_map,
    "unemp_reason": unemp_reason_index_map,
    "full_or_part_emp": full_or_part_emp_index_map,
    "tax_filer_stat": tax_filer_stat_index_map,
    "region_prev_res": region_prev_res_index_map,
    "state_prev_res": state_index_map,
    "det_hh_fam_stat": det_hh_index_map,
    "det_hh_summ": det_hh_index_map,
    "mig_chg_msa": mig_chg_index_map,
    "mig_chg_reg": mig_chg_index_map,
    "mig_move_reg": mig_chg_index_map,
    "mig_same": mig_same_index_map,
    "mig_prev_sunbelt": mig_prev_sunbelt_index_map,
    "fam_under_18": fam_under_18_index_map,
    'country_father': country_index_map,
    "country_mother": country_index_map,
    "country_self": country_index_map,
    "citizenship": citizenship_index_map,
    "vet_question": vet_question_index_map,
    "income_label": income_label_index
}

# each element is in the form of: feature_name:{input_dim, embedding_dim, tag}
embedding_dim_map = {
    "class_worker": (9, 4, "class_worker"),
    "education": (12, 6, "education"),
    "marital_stat": (5, 3, "marital_stat"),
    "major_ind_code": (24, 8, "major_ind_code"),
    "major_occ_code": (15, 6, "major_occ_code"),
    "race": (5, 3, "race"),
    "hisp_origin": (9, 5, "hisp_origin"),
    "age_index": (11, 6, "age_index"),
    "gender_index": (2, 1, "gender_index"),
    "union_member": (3, 2, "union_member"),
    "unemp_reason": (6, 4, "unemp_reason"),
    "full_or_part_emp": (8, 4, "full_or_part_emp"),
    "tax_filer_stat": (6, 4, "tax_filer_stat"),
    "region_prev_res": (6, 4, "region_prev_res"),
    "state_prev_res": (24, 8, "state_prev_res"),
    "det_hh_fam_stat": (44, 10, "det_hh"),
    "det_hh_summ": (44, 10, "det_hh"),
    # "mig_chg_msa": (19, 6, "mig_chg"),
    # "mig_chg_reg": (19, 6, "mig_chg"),
    # "mig_move_reg": (19, 6, "mig_chg"),
    "mig_chg_msa": (19, 6, "mig_chg_msa"),
    "mig_chg_reg": (19, 6, "mig_chg_reg"),
    "mig_move_reg": (19, 6, "mig_move_reg"),
    "mig_same": (4, 3, "mig_same"),
    "mig_prev_sunbelt": (4, 3, "mig_prev_sunbelt"),
    "fam_under_18": (5, 3, "fam_under_18"),
    # "country_father": (6, 4, "country"),
    # "country_mother": (6, 4, "country"),
    # "country_self": (6, 4, "country"),
    "country_father": (6, 4, "country_father"),
    "country_mother": (6, 4, "country_mother"),
    "country_self": (6, 4, "country_self"),
    "citizenship": (5, 3, "citizenship"),
    "vet_question": (3, 2, "vet_question"),
    "vet_benefits": (3, 2, "vet_benefits"),
    "own_or_self": (3, 2, "own_or_self")
}

if __name__ == "__main__":
    for key, val in cate_to_index_map.items():
        num_values = len(set(val.values()))
        print(key, num_values)

    print("#" * 10)
    for key, val in feature_group_map.items():
        group_emb_dim = 0
        for feat in val:
            tuple_v = embedding_dim_map[feat]
            emb_dim = tuple_v[1]
            group_emb_dim += emb_dim
        print(key, group_emb_dim)
