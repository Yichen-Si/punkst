# --------- Experimental: using cell centers to promote nuclei specific factors

# ─── Configurable parameters ────
GITPATH      := {GITPATH}
DATADIR      := {DATADIR}
TMPDIR       := {TMPDIR}
TRANSCRIPTS  := {TRANSCRIPTS}
ICOL_X       := {ICOL_X}
ICOL_Y       := {ICOL_Y}
ICOL_FEATURE := {ICOL_FEATURE}
ICOL_COUNT   := {ICOL_COUNT}
SKIP         := {SKIP}
EXCLUDE_FEATURE_REGEX := {EXCLUDE_FEATURE_REGEX}

THREADS      := {THREADS}
SEED         := {SEED}
TILESIZE     := {TILESIZE}
HEXGRIDS     := {HEXGRIDS}
TOPICS       := {TOPICS}
NEPOCHS	     := {NEPOCHS}
PIXHEX       := {PIXHEX}
NMOVE        := {NMOVE}
RES          := {RES}
SCALE        := {SCALE}

CELLCOORD    := {CELLCOORD}
NUCLEIRADIUS := {NUCLEIRADIUS}

NUR          := $(NUCLEIRADIUS)
PUNKST       := $(GITPATH)/bin/punkst
TILETSV      := $(DATADIR)/transcripts.tiled.tsv
TILEINDEX    := $(DATADIR)/transcripts.tiled.index
FEATURES     := $(DATADIR)/transcripts.tiled.features.tsv

TAB := $(shell printf '\t')

# ─── Compute the full list of final report files ──────────────────────────
REPORTS := \
  $(foreach H,$(HEXGRIDS),\
    $(foreach K,$(TOPICS),\
      $(DATADIR)/hex_$(H)_$(NUR).k$(K).pixel.html \
      $(DATADIR)/hex_$(H)_$(NUR).k$(K).pixel.png \
    )\
  )

.PHONY: all clean
all: $(REPORTS)

# ─── Step 1: bag pixels → tiles.tsv + tiles.index ─────────────────────
$(DATADIR)/prepare.stamp: $(TRANSCRIPTS)
	@echo "[1/6] pts2tiles → $@"
	 $(PUNKST) pts2tiles \
	  --in-tsv $(TRANSCRIPTS) \
	  --icol-x $(ICOL_X) --icol-y $(ICOL_Y) \
	  --icol-feature $(ICOL_FEATURE) --icol-int $(ICOL_COUNT) --skip $(SKIP) \
	  --tile-size $(TILESIZE) \
	  --out-prefix $(DATADIR)/transcripts.tiled \
	  --temp-dir $(TMPDIR) --threads $(THREADS)
	touch $(DATADIR)/prepare.stamp

# ─── Step 2: for each H in HEXGRIDS → hex_H.txt ────────────
define HEX_RULE
$(DATADIR)/hex_$(1)_$(NUR).txt: \
    $(DATADIR)/prepare.stamp \
	$(CELLCOORD)
	@echo "[2/6] tiles2hex (hex=$(1)) → $$@"
	 $(PUNKST) tiles2hex \
	  --in-tsv $(TILETSV) \
	  --in-index $(TILEINDEX) \
	  --feature-dict $(FEATURES) \
	  --anchor-files $(CELLCOORD) \
	  --radius $(NUCLEIRADIUS) \
	  --icol-x $(ICOL_X) --icol-y $(ICOL_Y) \
	  --icol-feature $(ICOL_FEATURE) \
	  --icol-int $(ICOL_COUNT) \
	  --hex-grid-dist $(1) \
	  --min-count 50 \
	  --out $(DATADIR)/hex_$(1)_$(NUR).txt \
	  --randomize \
	  --temp-dir $(TMPDIR) --threads $(THREADS)
endef
$(foreach H,$(HEXGRIDS),$(eval $(call HEX_RULE,$H)))

# ─── Step 3: for each (K,H) → model.tsv + color.rgb.tsv ──────────────
define LDA_MODEL_RULE
$(DATADIR)/hex_$(1)_$(NUR).k$(2).model.stamp: \
    $(DATADIR)/hex_$(1)_$(NUR).txt
	@echo "[3/6] lda4hex (K=$(2), hex=$(1)) → $$@"
	 $(PUNKST) lda4hex \
	  --in-data $(DATADIR)/hex_$(1)_$(NUR).txt \
	  --in-meta $(DATADIR)/hex_$(1)_$(NUR).json \
	  --features $(FEATURES) \
	  --exclude-feature-regex $(EXCLUDE_FEATURE_REGEX) \
	  --n-topics $(2) --n-epochs $(NEPOCHS) --sort-topics \
	  --min-count-per-feature 100 --min-count-train 50 \
	  --out-prefix $(DATADIR)/hex_$(1)_$(NUR).k$(2) --transform \
	  --threads $(THREADS) --seed $(SEED)
	touch $(DATADIR)/hex_$(1)_$(NUR).k$(2).model.stamp
endef
$(foreach H,$(HEXGRIDS),\
  $(foreach K,$(TOPICS),\
    $(eval $(call LDA_MODEL_RULE,$H,$K))\
  )\
)

define COLOR_RULE
$(DATADIR)/hex_$(1)_$(NUR).k$(2).color.rgb.tsv: \
    $(DATADIR)/hex_$(1)_$(NUR).k$(2).model.stamp
	@echo "[3/6] color_helper (K=$(2), hex=$(1)) → $$@"
	python $(GITPATH)/ext/py/color_helper.py \
	  --input $(DATADIR)/hex_$(1)_$(NUR).k$(2).results.tsv \
	  --output $(DATADIR)/hex_$(1)_$(NUR).k$(2).color \
	  --skip-columns layer
endef
$(foreach H,$(HEXGRIDS),\
  $(foreach K,$(TOPICS),\
    $(eval $(call COLOR_RULE,$H,$K)) \
  )\
)

# ─── Step 4: pixel-decode → binary + .pseudobulk.tsv ────────────────────
define PIXEL_RULE
$(DATADIR)/hex_$(1)_$(NUR).k$(2).pixel.decode.stamp: \
    $(DATADIR)/hex_$(1)_$(NUR).k$(2).model.stamp\
    $(DATADIR)/prepare.stamp \
    $(CELLCOORD)
	@echo "[4/6] pixel-decode (K=$(2), hex=$(1)) → $$@"
	 $(PUNKST) pixel-decode \
	  --model $(DATADIR)/hex_$(1)_$(NUR).k$(2).model.tsv \
	  --in-tsv $(TILETSV) --in-index $(TILEINDEX) --anchor $(CELLCOORD) \
	  --icol-x $(ICOL_X) --icol-y $(ICOL_Y) \
	  --icol-feature $(ICOL_FEATURE) --icol-val $(ICOL_COUNT) \
	  --hex-grid-dist $(PIXHEX) --n-moves $(NMOVE) --pixel-res $(RES) \
	  --min-init-count 20 --output-binary \
	  --out-pref $(DATADIR)/hex_$(1)_$(NUR).k$(2).pixel \
	  --temp-dir $(TMPDIR) --threads $(THREADS) --seed $(SEED)
	touch $(DATADIR)/hex_$(1)_$(NUR).k$(2).pixel.decode.stamp
endef
$(foreach H,$(HEXGRIDS),\
  $(foreach K,$(TOPICS),\
    $(eval $(call PIXEL_RULE,$H,$K))\
  )\
)

# ─── Step 5: draw-pixel-factors → .png ───────────────────────────
define DRAW_RULE
$(DATADIR)/hex_$(1)_$(NUR).k$(2).pixel.png: \
    $(DATADIR)/hex_$(1)_$(NUR).k$(2).pixel.decode.stamp \
    $(DATADIR)/hex_$(1)_$(NUR).k$(2).color.rgb.tsv
	@echo "[5/6] draw-pixel-factors (K=$(2), hex=$(1)) → $$@"
	 $(PUNKST) draw-pixel-factors \
	  --in $(DATADIR)/hex_$(1)_$(NUR).k$(2).pixel \
	  --binary \
	  --in-color $(DATADIR)/hex_$(1)_$(NUR).k$(2).color.rgb.tsv \
	  --out $$@ \
	  --scale $(SCALE)
endef
$(foreach H,$(HEXGRIDS),\
  $(foreach K,$(TOPICS),\
    $(eval $(call DRAW_RULE,$H,$K))\
  )\
)

# ─── Step 6: de-chisq + factor_report → final .report ───────────────
define DE_RULE
$(DATADIR)/hex_$(1)_$(NUR).k$(2).pixel.de_bulk.tsv: \
    $(DATADIR)/hex_$(1)_$(NUR).k$(2).pixel.decode.stamp
	@echo "[6/6] de-chisq (K=$(2), hex=$(1)) → $$@"
	$(PUNKST) de-chisq \
	  --input $(DATADIR)/hex_$(1)_$(NUR).k$(2).pixel.pseudobulk.tsv \
	  --out $(DATADIR)/hex_$(1)_$(NUR).k$(2).pixel.de_bulk.tsv \
	  --threads $(THREADS)
endef
$(foreach H,$(HEXGRIDS),\
  $(foreach K,$(TOPICS),\
    $(eval $(call DE_RULE,$H,$K))\
  )\
)

define REPORT_RULE
$(DATADIR)/hex_$(1)_$(NUR).k$(2).pixel.html: \
    $(DATADIR)/hex_$(1)_$(NUR).k$(2).pixel.decode.stamp \
	$(DATADIR)/hex_$(1)_$(NUR).k$(2).pixel.de_bulk.tsv \
    $(DATADIR)/hex_$(1)_$(NUR).k$(2).color.rgb.tsv
	@echo "[6/6] factor_report (K=$(2), hex=$(1)) → $$@"
	python $(GITPATH)/ext/py/factor_report.py \
	  --de $(DATADIR)/hex_$(1)_$(NUR).k$(2).pixel.de_bulk.tsv \
	  --pseudobulk $(DATADIR)/hex_$(1)_$(NUR).k$(2).pixel.pseudobulk.tsv \
	  --feature_label Feature \
	  --color_table $(DATADIR)/hex_$(1)_$(NUR).k$(2).color.rgb.tsv \
	  --output_pref $(DATADIR)/hex_$(1)_$(NUR).k$(2).pixel
endef
$(foreach H,$(HEXGRIDS),\
  $(foreach K,$(TOPICS),\
    $(eval $(call REPORT_RULE,$H,$K))\
  )\
)

# ─── Clean up all generated files ──────────────────────────────────
clean:
	rm -rf $(TMPDIR) \
	       $(DATADIR)/transcripts.tiled.* \
	       $(DATADIR)/hex_*_*.txt \
	       $(DATADIR)/hex_*_*.json \
	       $(DATADIR)/hex_*_*.k*
