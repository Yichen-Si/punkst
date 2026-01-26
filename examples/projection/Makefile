# --- Configurable parameters (these get filled in by your Python script) ---
GITPATH      := {GITPATH}
DATADIR      := {DATADIR}
TMPDIR       := {TMPDIR}
TRANSCRIPTS  := {TRANSCRIPTS}
ICOL_X       := {ICOL_X}
ICOL_Y       := {ICOL_Y}
ICOL_FEATURE := {ICOL_FEATURE}
ICOL_COUNT   := {ICOL_COUNT}
SKIP         := {SKIP}

THREADS      := {THREADS}
SEED         := {SEED}
TILESIZE     := {TILESIZE}
PIXHEX       := {PIXHEX}
NMOVE        := {NMOVE}
RES          := {RES}
SCALE        := {SCALE}

MODEL_TSV    := {MODEL_TSV}
OUT_PREF     := {OUT_PREF}
COLOR_TABLE  := {COLOR_TABLE}

PUNKST       := $(GITPATH)/bin/punkst
TILETSV      := $(DATADIR)/transcripts.tiled.tsv
TILEINDEX    := $(DATADIR)/transcripts.tiled.index
COORD_RANGE  := $(DATADIR)/transcripts.tiled.coord_range.tsv

# --- Compute the full list of final report files ---
REPORTS := \
  $(OUT_PREF).pixel.html \
  $(OUT_PREF).pixel.png

.PHONY: all clean clean_all
all: $(REPORTS)

# --- Step 1: bag pixels -> tiles.tsv + tiles.index ---
$(DATADIR)/prepare.stamp: $(TRANSCRIPTS)
	@echo "[1/4] pts2tiles -> $@"
	 $(PUNKST) pts2tiles \
	  --in-tsv $(TRANSCRIPTS) \
	  --icol-x $(ICOL_X) --icol-y $(ICOL_Y) \
	  --icol-feature $(ICOL_FEATURE) --icol-int $(ICOL_COUNT) --skip $(SKIP) \
	  --tile-size $(TILESIZE) \
	  --out-prefix $(DATADIR)/transcripts.tiled \
	  --temp-dir $(TMPDIR) --threads $(THREADS)
	touch $(DATADIR)/prepare.stamp

# --- Step 2: pixel-decode -> .tsv + .pseudobulk.tsv ---
$(OUT_PREF).pixel.decode.stamp: \
    $(DATADIR)/prepare.stamp \
    $(MODEL_TSV)
	@echo "[2/4] pixel-decode -> $@"
	 $(PUNKST) pixel-decode \
	  --model $(MODEL_TSV) \
	  --in-tsv $(TILETSV) --in-index $(TILEINDEX) \
	  --icol-x $(ICOL_X) --icol-y $(ICOL_Y) \
	  --icol-feature $(ICOL_FEATURE) --icol-val $(ICOL_COUNT) \
	  --hex-grid-dist $(PIXHEX) --n-moves $(NMOVE) --pixel-res $(RES) \
	  --min-init-count 20 \
	  --out-pref $(OUT_PREF).pixel --output-binary \
	  --temp-dir $(TMPDIR) --threads $(THREADS) --seed $(SEED)
	$(PUNKST) tile-op --reorganize --binary --in $(OUT_PREF).pixel --out $(OUT_PREF).pixel.reorg
	mv $(OUT_PREF).pixel.reorg.bin $(OUT_PREF).pixel.bin
	mv $(OUT_PREF).pixel.reorg.index $(OUT_PREF).pixel.index
	touch $(OUT_PREF).pixel.decode.stamp

# --- Step 3: draw-pixel-factors -> png ---
$(OUT_PREF).pixel.png: \
    $(OUT_PREF).pixel.decode.stamp \
    $(COLOR_TABLE)
	@echo "[3/4] draw-pixel-factors -> $@"
	 $(PUNKST) draw-pixel-factors \
	  --in $(OUT_PREF).pixel --binary \
	  --in-color $(COLOR_TABLE) \
	  --out $@ \
	  --range $(COORD_RANGE) --scale $(SCALE)

# --- Step 4: de_bulk + factor_report -> final .report ---
$(OUT_PREF).pixel.de_bulk.tsv: \
    $(OUT_PREF).pixel.decode.stamp
	@echo "[4/4] de_bulk -> $@"
	$(PUNKST) de-chisq \
	  --input $(OUT_PREF).pixel.pseudobulk.tsv \
	  --out $(OUT_PREF).pixel.de_bulk.tsv \
	  --threads $(THREADS)

$(OUT_PREF).pixel.html: \
    $(OUT_PREF).pixel.decode.stamp \
    $(OUT_PREF).pixel.de_bulk.tsv \
    $(COLOR_TABLE)
	@echo "[4/4] factor_report -> $@"
	python $(GITPATH)/ext/py/factor_report.py \
	  --de $(OUT_PREF).pixel.de_bulk.tsv \
	  --de_neighbor $(OUT_PREF).pixel.de_bulk.1vsNeighbors.tsv \
	  --pseudobulk $(OUT_PREF).pixel.pseudobulk.tsv \
	  --feature_label Feature \
	  --color_table $(COLOR_TABLE) \
	  --output_pref $(OUT_PREF).pixel

# --- Clean up parameter specific files ---
clean:
	rm -rf $(TMPDIR) \
	       $(OUT_PREF).pixel*
clean_all:
	rm -rf $(TMPDIR) \
	       $(DATADIR)/transcripts.tiled.* \
	       $(OUT_PREF).pixel*
