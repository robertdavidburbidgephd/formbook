#!/bin/bash
cat extracted/SFF_*.psv | cut -d\| -f2,6 | awk --field-separator='|' '$2=="c" { print $1 }' - | sort -u > dict/vals/chase_courses.psv

