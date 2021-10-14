PDVARS += --verbose
PDVARS += --highlight-style=pygments # the default theme

# Optional filters, must be installed to use:
#PDVARS += --filter pandoc-include
#PDVARS += --filter pandoc-codeblock-include
#PDVARS += --filter pandoc-imagine
#PDVARS += --filter pandoc-crossref

.PHONY: help
help:
	@echo "try running 'make template.pdf'"

.PHONY: has-%
has-%:
	@command -v $* >/dev/null || ( \
		echo "ERROR: Command '$*' not found! Make sure it is installed and available in PATH"; \
		false; \
	) >&2

%.pdf: %.md Makefile has-pandoc
	pandoc -i $< ${PDVARS} -o $@ #--pdf-engine=pdflatex

%.tex: %.md Makefile has-pandoc
	pandoc -i $< ${PDVARS} -o $@ --standalone

%.html: %.md Makefile has-pandoc
	pandoc -i $< ${PDVARS} -o $@ --katex --standalone --self-contained
