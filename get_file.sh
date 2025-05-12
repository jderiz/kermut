
# Available files:
# Data	Size (unzipped)	Filename
# DMS benchmark - Substitutions	1.0GB	DMS_ProteinGym_substitutions.zip
# DMS benchmark - Indels	200MB	DMS_ProteinGym_indels.zip
# Zero-shot DMS Model scores - Substitutions	4.4GB	zero_shot_substitutions_scores.zip
# Zero-shot DMS Model scores - Indels	313MB	zero_shot_indels_scores.zip
# Supervised DMS Model scores - Substitutions	3.3GB	DMS_supervised_substitutions_scores.zip
# Supervised DMS Model scores - Indels	215MB	DMS_supervised_indels_scores.zip
# Multiple Sequence Alignments (MSAs) for DMS assays	5.2GB	DMS_msa_files.zip
# Redundancy-based sequence weights for DMS assays	200MB	DMS_msa_weights.zip
# Predicted 3D structures from inverse-folding models	84MB	ProteinGym_AF2_structures.zip
# Clinical benchmark - Substitutions	123MB	clinical_ProteinGym_substitutions.zip
# Clinical benchmark - Indels	2.8MB	clinical_ProteinGym_indels.zip
# Clinical MSAs	17.8GB	clinical_msa_files.zip
# Clinical MSA weights	250MB	clinical_msa_weights.zip
# Clinical Model scores - Substitutions	0.9GB	zero_shot_clinical_substitutions_scores.zip
# Clinical Model scores - Indels	0.7GB	zero_shot_clinical_indels_scores.zip
# CV folds - Substitutions - Singles	50M	cv_folds_singles_substitutions.zip
# CV folds - Substitutions - Multiples	81M	cv_folds_multiples_substitutions.zip
# CV folds - Indels	19MB	cv_folds_indels.zip


# defualt values
VERSION="v1.3"
FILENAME="DMS_ProteinGym_substitutions.zip"


function get_file() {
    FILENAME=$1
    curl -o ${FILENAME} https://marks.hms.harvard.edu/proteingym/ProteinGym_${VERSION}/${FILENAME}
    unzip ${FILENAME} && rm ${FILENAME}
}



# autocompletion for files
_get_file() {
    local cur prev opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    opts="DMS_ProteinGym_substitutions.zip DMS_ProteinGym_indels.zip zero_shot_substitutions_scores.zip zero_shot_indels_scores.zip DMS_supervised_substitutions_scores.zip DMS_supervised_indels_scores.zip DMS_msa_files.zip DMS_msa_weights.zip ProteinGym_AF2_structures.zip clinical_ProteinGym_substitutions.zip clinical_ProteinGym_indels.zip clinical_msa_files.zip clinical_msa_weights.zip zero_shot_clinical_substitutions_scores.zip zero_shot_clinical_indels_scores.zip cv_folds_singles_substitutions.zip cv_folds_multiples_substitutions.zip cv_folds_indels.zip"
    COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
}





# for all files provided as arguments, call get_file
#

for FILENAME in $@; do
    get_file ${FILENAME}.zip
done



complete -F _get_file get_file