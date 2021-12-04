#!/usr/bin/env perl

use strict;
use locale;

use utf8;
binmode STDIN, ":utf8";
binmode STDOUT, ":utf8";

my $ligne;
while ($ligne=<>)
{
	chomp($ligne);
	if(($ligne eq "") or ($ligne eq "<P>"))
	{
		print(STDOUT "\n");
	}
	else
	{
		print(STDOUT "$ligne\n");
	}
};
