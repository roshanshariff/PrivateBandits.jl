#!/bin/bash

basedir="$(dirname "$0")"
filename="${basedir}/private-bandits.pdf"
pdfinfo="${basedir}/pdfinfo.txt"
output1="${basedir}/nips_submission.pdf"
output2="${basedir}/nips_supplementary.pdf"

pdftk "$filename" dump_data output "$pdfinfo"
startsupp="$(awk -F': ' -- '/BookmarkTitle: Supplementary Material/,0 { if ($1 == "BookmarkPageNumber") { print $2; exit } }' "$pdfinfo")"
graveyard="$(awk -F': ' -- '/BookmarkTitle: Graveyard/,0 { if ($1 == "BookmarkPageNumber") { print $2; exit } }' "$pdfinfo")"

pdftk "$filename" cat "1-$((startsupp-1))" output "${basedir}/nips_submit.pdf"
pdftk "$filename" cat "${startsupp}-$((graveyard-1))" output "${basedir}/nips_supplementary.pdf"
