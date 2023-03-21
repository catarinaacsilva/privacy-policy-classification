#!/bin/bash

# Use TMPDIR if defined or fallback to /tmp
TMP="${TMPDIR:-/tmp}"

# Download the dataset into the TMP folder
curl https://usableprivacy.org/static/data/OPP-115_v1_0.zip --progress-bar --output $TMP/OPP-115_v1_0.zip

# Unzip the dataset
unzip -o -q $TMP/OPP-115_v1_0.zip -d $TMP

# Create the necessary folders
[ ! -d "sanitized_policies" ] &&  mkdir -p "sanitized_policies"
[ ! -d "annotations" ] && mkdir -p "annotations"
[ ! -d "corpus" ] && mkdir -p "corpus"

# Move the sanitized policies from the TMP folder to the working folder
mv -f $TMP/OPP-115/sanitized_policies/*.html sanitized_policies/

# Move the annotations from the TMP folder to the working folder
mv -f $TMP/OPP-115/annotations/*.csv annotations/

echo -e "Done."