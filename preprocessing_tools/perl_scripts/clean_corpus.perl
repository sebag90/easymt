#!/usr/bin/env perl

# This file is part of moses.  Its use is licensed under the GNU Lesser General
# Public License version 2.1 or, at your option, any later version.


# clean-corpus-n.perl CORPUS L1 L2 OUT MIN MAX
# For example: clean-corpus-n.perl raw de en clean 1 50 takes the corpus files raw.de and raw.en, 
# deletes lines longer than 50, and creates the output files clean.de and clean.en. 



use warnings;
use strict;
use Getopt::Long;
my $help;
my $lc = 0; # lowercase the corpus?
my $ignore_ratio = 0;
my $ignore_xml = 0;
my $enc = "utf8"; # encoding of the input and output files
    # set to anything else you wish, but I have not tested it yet
my $max_word_length = 1000; # any segment with a word (or factor) exceeding this length in chars
    # is discarded; motivated by symal.cpp, which has its own such parameter (hardcoded to 1000)
    # and crashes if it encounters a word that exceeds it	
my $ratio = 9;

GetOptions(
  "help" => \$help,
  "lowercase|lc" => \$lc,
  "encoding=s" => \$enc,
  "ratio=f" => \$ratio,
  "ignore-ratio" => \$ignore_ratio,
  "ignore-xml" => \$ignore_xml,
  "max-word-length|mwl=s" => \$max_word_length
) or exit(1);

if (scalar(@ARGV) < 2 || $help) {
    print "syntax: clean-corpus-n.perl [-ratio n] corpus l1 l2 clean-corpus min max [lines retained file]\n";
    exit;
}

my $min = $ARGV[0];
my $max = $ARGV[1];


my $linesRetainedFile = "";
if (scalar(@ARGV) > 2) {
	$linesRetainedFile = $ARGV[6];
	open(LINES_RETAINED,">$linesRetainedFile") or die "Can't write $linesRetainedFile";
}

#print STDERR "clean-corpus.perl: processing $corpus.$l1 & .$l2 to $out, cutoff $min-$max, ratio $ratio\n";
binmode(STDIN, ":utf8");
binmode(STDOUT, ":utf8");

my $innr = 0;
my $outnr = 0;
my $factored_flag;
while(<STDIN>) {
  chomp;
  my @spl = split('\t', $_);
  my $f = (@spl)[0];
  my $e = (@spl)[1];
 
  $innr++;
  print STDERR "." if $innr % 10000 == 0;
  print STDERR "($innr)" if $innr % 100000 == 0;


  if ($innr == 1) {
    $factored_flag = ($e =~ /\|/ || $f =~ /\|/);
  }

  #if lowercasing, lowercase
  if ($lc) {
    $e = lc($e);
    $f = lc($f);
  }

  $e =~ s/\|//g unless $factored_flag;
  $e =~ s/\s+/ /g;
  $e =~ s/^ //;
  $e =~ s/ $//;
  $f =~ s/\|//g unless $factored_flag;
  $f =~ s/\s+/ /g;
  $f =~ s/^ //;
  $f =~ s/ $//;
  next if $f eq '';
  next if $e eq '';

  my $ec = &word_count($e);
  my $fc = &word_count($f);
  next if $ec > $max;
  next if $fc > $max;
  next if $ec < $min;
  next if $fc < $min;
  next if !$ignore_ratio && $ec/$fc > $ratio;
  next if !$ignore_ratio && $fc/$ec > $ratio;
  # Skip this segment if any factor is longer than $max_word_length
  my $max_word_length_plus_one = $max_word_length + 1;
  next if $e =~ /[^\s\|]{$max_word_length_plus_one}/;
  next if $f =~ /[^\s\|]{$max_word_length_plus_one}/;

  # An extra check: none of the factors can be blank!
  die "There is a blank factor in : $f"
    if $f =~ /[ \|]\|/;
  die "There is a blank factor in : $e"
    if $e =~ /[ \|]\|/;

  $outnr++;
  print $f."\t".$e."\n";
 # print EO $e."\n";

  if ($linesRetainedFile ne "") {
	print LINES_RETAINED $innr."\n";
  }
}



#print STDERR "Input sentences: $innr  Output sentences:  $outnr\n";

sub word_count {
  my ($line) = @_;
  if ($ignore_xml) {
    $line =~ s/<\S[^>]*\S>/ /g;
    $line =~ s/\s+/ /g;
    $line =~ s/^ //g;
    $line =~ s/ $//g;
  }
  my @w = split(/ /,$line);
  return scalar @w;
}
