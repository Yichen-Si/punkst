# --------- Prepare Vizgen MERSCOPE data

# ─── Configurable parameters ────
RAW_TX   := {RAW_TX}
RAW_META := {RAW_META}
MICRONS_PER_PIXEL := {MICRONS_PER_PIXEL}
DATADIR      := {DATADIR}

TRANSCRIPTS  := $(DATADIR)/transcripts.tsv
CELLCOORD    := $(DATADIR)/cell_centers.tsv

TAB := $(shell printf '\t')

# ─── Step 0. Prepare‐data ─────────────────────────────────────────────────
.PHONY: all
all: $(CELLCOORD) $(TRANSCRIPTS)

$(CELLCOORD):
	cut -d',' -f 1-3,7-8 $(RAW_META) | tail -n +2 | awk -F',' -v OFS="$(TAB)" \
		-v mu=$(MICRONS_PER_PIXEL) \
	    '{{printf "%.2f\t%.2f\t%s_%s\t%s\n", mu * $$4, mu * $$5, $$1, $$2, $$3}}' > $(CELLCOORD)

$(TRANSCRIPTS):
	@echo "Extracting transcripts..."
	  awk -F',' -v mu=$(MICRONS_PER_PIXEL) '\
	  NR==1{{gsub(/"/, "", $$0); print "#" $$3, $$4, $$8, "count", $$7, "cell_ID", $$9 }}\
	  NR>1{{gsub(/"/, "", $$8); gsub(/"/, "", $$9); printf "%.2f\t%.2f\t%s\t%d\t%d\t%s_%s\t%s\n", mu*$$3, mu*$$4, $$8, 1, $$7, $$1, $$2, $$9 }} ' $(RAW_TX) > $@
