#!/bin/bash
# source scripts/courses.sh
#cat data/prep/SFF_2024.psv | cut -d\| -f2,6 | awk -F \| '$2=="" { print $1 }' - | uniq | sort -u | awk '$2==""' -
# doesn't give IRE/NI courses ?? CHK
cut -d\| -f2,6 data/extracted/SFF_2024.psv | sort -u | uniq | awk -F\| '$2=="" {print $1}' - > artefacts/flat_courses.txt

