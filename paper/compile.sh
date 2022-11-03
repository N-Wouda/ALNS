#!/usr/bin/env bash

# Help text

description="A bash script for compiling JOSS and JOSE papers locally";

doc="compile [-j <journal>] [-d <doi>] [-v <volume>] [-i <issue>] [-p <page>]
        [-s <submitted>] [-u <published>] [-r <review>] [-g <repository>] [-a <archive>] FILE

Compile a markdown FILE to a JOSS or JOSE paper.

Options:
  -j JOURNAL      The journal the paper is being compiled for, either joss or jose, defaults to joss.
  -d DOI          The DOI of the submitted paper, defaults to an empty string.
  -v VOLUME       The journal volume the paper was published in, defaults to an empty string.
  -i ISSUE        The journal issue the paper was published in, defaults to an empty string.
  -p PAGE         The journal issue page the paper was published in, defaults to an empty string.
  -s SUBMITTED    The date the paper was submitted, defaults to an empty string.
  -u PUBLISHED    The date the paper was published, defaults to an empty string.
  -r REVIEW       The review issue URL, defaults to an empty string.
  -g REPOSITORY   The URL of the code repository, defaults to an empty string.
  -a ARCHIVE      The DOI of the code archive, defaults to an empty string.
  -h              Displays this help text and exit."

# Default arguments

journal="joss"
doi=""
volume=""
issue=""
page=""
submitted=""
published=""
review=""
repository=""

# Get command line arguments

while getopts ":j:d:v:i:p:s:u:r:g:a:h" opt; do
	case $opt in
		j)
			journal=$OPTARG;;
		d)
			doi=$OPTARG;;
		v)
			volume=$OPTARG;;
		i)
			issue=$OPTARG;;
		p)
			page=$OPTARG;;
		s)
			submitted=$OPTARG;;
		u)
			published=$OPTARG;;
		r)
			review=$OPTARG;;
		g)
			repository=$OPTARG;;
		a)
			archive=$OPTARG;;
		h)
			echo $description;
			echo;
			echo "$doc";
			exit 0;;
		\?)
			echo "Invalid option: -$OPTARG.";
			echo;
			echo "$doc";
			exit 1;;
		:)
			echo "Option -$OPTARG requires an argument.";
			echo;
			echo "$doc";
			exit 1;;
	esac;
done;

paper=${@:$OPTIND:1}

if [ ! "$paper" ]; then
	echo "FILE not specified."
	echo;
	echo "$doc"
	exit 1;
fi;

# Check if pandoc and pandoc-citeproc are installed

missing=""

if [ ! "$(which pandoc)" ]; then
	missing="$missing pandoc";
fi;

if [ ! "$(which pandoc-citeproc)" ]; then
	missing="$missing pandoc-citeproc";
fi;

if [ ! "$(which xelatex)" ]; then
	missing="$missing xelatex";
fi;

if [ "$missing" ]; then
	if [ $(echo "$missing" | wc -w) -eq 2 ]; then
		echo "Please install $(echo $missing | sed 's/ / and /').";
	else
		echo "Please install $(echo $missing | sed 's/ /, /g' | sed 's/\(.*\) /\1 and /').";
	fi;
	exit 1;
fi;

# Set journal specific variables

outfile="${paper%.md}.pdf"
logo="${journal}-logo.png"
templateurl="https://raw.githubusercontent.com/openjournals/whedon/master/resources/joss/latex.template"
cslurl="https://raw.githubusercontent.com/openjournals/whedon/master/resources/apa.csl"
logourl="https://github.com/openjournals/whedon/blob/master/resources/${journal}/logo.png?raw=true"

# Set journal full name

if [ $journal == "joss" ]; then
	journalname="Journal of Open Source Software";
else
	journalname="Journal of Open Source Education";
fi;

# Download logo and latex template files

if [ ! -f latex.template ]; then
	wget "$templateurl";
fi;

if [ ! -f apa.csl ]; then
	wget "$cslurl";
fi;

if [ ! -f $logo ]; then
	wget -O $logo "$logourl";
fi;

# Collect necessary paper metadata

year=$(grep "date: " $paper | cut -d " " -f 4 -)
bib=$(grep "bibliography: " $paper | cut -d " " -f 2 -)

title=$( \
	grep "title: " $paper | \
	cut -d " " -f 2- - | \
	sed 's/^"\(.*\)"$/\1/' | \
	sed "s/^'\(.*\)'$/\1/")

name=$(grep -m 1 "name: " $paper | rev | cut -d " " -f 1 - | rev)

# Compile the paper

pandoc \
	--filter pandoc-citeproc \
	--bibliography=$bib \
	--template=latex.template \
	-V logo_path=$logo \
	-V citation_author=$name \
	-V year=$year \
	-V footnote_paper_title="$title" \
	-V journal_name="$journalname" \
	-V formatted_doi="$doi" \
	-V archive_doi="https://doi.org/$archive" \
	-V review_issue_url=$review \
	-V repository=$repository \
	-V submitted=$submitted \
	-V published=$published \
	-V issue=$issue \
	-V volume=$volume \
	-V page=$page \
	-V graphics=true \
	--pdf-engine=xelatex \
	--from markdown+autolink_bare_uris \
	--csl=apa.csl \
	-s $paper \
	-o $outfile
