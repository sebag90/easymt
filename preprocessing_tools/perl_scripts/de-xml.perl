#!/usr/bin/perl -w


# perl de-xml.perl corpus l1 l2 output-name


use strict;


binmode(STDIN, ":utf8");
binmode(STDOUT, ":utf8");

my $i=0;

while(<STDIN>) {
  chomp;
  my @spl = split('\t', $_);
  my $f = (@spl)[0];
  my $e = (@spl)[1];

  # split stdin
  $i++;
  #chop($e); chop($f);
  next if ($e =~ /^<.+>$/) && ($f =~ /^<.+>$/);
  if (($e =~ /^<.+>$/) || ($f =~ /^<.+>$/)) {
    print STDERR "MISMATCH[$i]: $e <=> $f\n";
    next;
  }
  if (($e =~ /<.+>/) || ($f =~ /<.+>/)) {
      #print STDERR "TAGS IN TEXT, STRIPPING[$i]: $e <=> $f\n";
      $e =~ s/ *<[^>]+> */ /g;
      $e =~ s/^ +//;
      $e =~ s/ +$//;
      $f =~ s/ *<[^>]+> */ /g;
      $f =~ s/^ +//;
      $f =~ s/ +$//;
      #print STDERR "TAGS STRIPPED: $e <=> $f\n";
  }

  print $f."\t".$e."\n";
}
