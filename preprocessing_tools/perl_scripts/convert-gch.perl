#!/usr/bin/env perl

use strict;
use locale;

use utf8;
binmode STDIN, ":utf8";
binmode STDOUT, ":utf8";
binmode STDERR, ":utf8";

my $ligne;
my $name;
my @tab;
my $i;

while ($ligne=<>)
{
	chomp($ligne);
	if(($ligne eq "") or ($ligne eq "<P>"))
	{
		print(STDOUT ".EOP\n");
	}
	else
	{
		@tab=split(/ /,$ligne);
		for($i=0; $i <= $#tab; $i++)
		{
			print(STDOUT "$tab[$i]\n");
		};
		print(STDOUT ".EOS\n");
	}
};

print(STDOUT ".EOP\n");
