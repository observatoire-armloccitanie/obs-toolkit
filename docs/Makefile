# Construction de la documentation : `make html`
SPHINXBUILD ?= sphinx-build
SOURCEDIR   = source
BUILDDIR    = build

.PHONY: help html clean
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)"

html:
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)"

clean:
	rm -rf $(BUILDDIR) $(SOURCEDIR)/api/generated
