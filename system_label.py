# The OCT acquisition software on different systems behaves differently.
# There is an XML file stored with each UNP data file, and the way in
# which acquisition parameters are written to this file, and the logic
# of the acquisiton parameters, differs among systems. The way in which
# octoblob digests the XML file and uses the resulting parameters is system-
# dependent.
# Valid choices here are 'clinical_org' and 'eyepod'. Just comment out the
# one you don't want.

system_label = 'clinical_org'
# system_label = 'eyepod'
