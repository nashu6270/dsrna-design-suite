"""
dsRNA Design Suite — Streamlit Web App
FASTA → Sliding-window fragments → Score & rank → 500-DPI PNG figures
Authors : S Nanda · C Bose · A Nayak
"""
import io, json, re, warnings
from collections import Counter
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from Bio import SeqIO
from Bio.SeqUtils import gc_fraction

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.linewidth": 1.6, "savefig.bbox": "tight",
})

# =====================================================================
#  CORE LOGIC
# =====================================================================

class dsRNADesigner:
    def __init__(self, cds_sequence, gene_name, masked_regions=None, mask_tolerance=0.1):
        self.cds_sequence   = cds_sequence.upper()
        self.cds_length     = len(self.cds_sequence)
        self.gene_name      = gene_name
        self.masked_regions = masked_regions or []
        self.mask_tolerance = mask_tolerance
        self.fragments      = []

    def check_masked_overlap(self, start, end):
        wl = end - start; ov = 0
        ws1, we1 = start + 1, end
        for ms, me in self.masked_regions:
            o1, o2 = max(ws1, ms), min(we1, me)
            if o1 <= o2: ov += o2 - o1 + 1
        frac = ov / wl if wl else 0
        return frac <= self.mask_tolerance, frac

    def count_homopolymers(self, seq, min_len=6):
        r = {b: 0 for b in "ACGT"}
        for b in "ACGT":
            r[b] = len(re.findall(rf"{b}{{{min_len},}}", seq.upper()))
        return r

    def calc_dinuc(self, seq):
        s = re.sub("[^ACGT]", "", seq.upper())
        d = [s[i:i+2] for i in range(len(s)-1)]
        t = len(d); c = Counter(d)
        return {k: v/t for k, v in c.items()} if t else {}

    def calc_trinuc(self, seq):
        s = re.sub("[^ACGT]", "", seq.upper())
        d = [s[i:i+3] for i in range(len(s)-2)]
        t = len(d); c = Counter(d)
        return {k: v/t for k, v in c.items()} if t else {}

    def count_kmer_dup(self, seq, k=21):
        s = re.sub("[^ACGT]", "", seq.upper())
        if len(s) < k: return 0, 0
        km = [s[i:i+k] for i in range(len(s)-k+1)]
        c = Counter(km); tot = len(km)
        dup = sum(1 for v in c.values() if v > 1)
        return tot, dup

    def annotate(self, fseq, fid, s1, e1, otype, mfrac):
        aseq    = re.sub("[^ACGT]", "", fseq.upper())
        gc      = gc_fraction(aseq) if aseq else 0.0
        hp6     = self.count_homopolymers(fseq, 6)
        hp8     = self.count_homopolymers(fseq, 8)
        t21, d21= self.count_kmer_dup(fseq, 21)
        kdr     = d21 / t21 if t21 else 0
        issues  = []
        if sum(hp8.values()) > 0: issues.append("homopolymer_8plus")
        if kdr > 0.1:              issues.append("high_kmer_duplication")
        if mfrac > 0:              issues.append("masked_region_overlap")
        if gc < 0.3 or gc > 0.7:  issues.append("extreme_gc_content")
        return {
            "fragment_id": fid, "gene_name": self.gene_name,
            "start_pos_1based": s1, "end_pos_1based": e1,
            "length": len(fseq), "overlap_type": otype, "sequence": fseq,
            "gc_content": gc, "gc_percentage": gc * 100,
            "dinuc_composition_json": json.dumps(self.calc_dinuc(fseq)),
            "trinuc_composition_json": json.dumps(self.calc_trinuc(fseq)),
            "homopolymer_runs_6plus": sum(hp6.values()),
            "homopolymer_runs_8plus": sum(hp8.values()),
            "flagged_homopolymers": sum(hp8.values()) > 0,
            "homopolymer_details": json.dumps(hp6),
            "total_21mers": t21, "duplicate_21mers": d21,
            "kmer_duplication_rate": kdr,
            "masked_overlap_fraction": mfrac,
            "masked_overlap_percentage": mfrac * 100,
            "design_issues_str": ";".join(issues) if issues else "none",
        }

    def generate_fragments(self, window_lengths, step_sizes,
                            do_overlap=True, do_nonoverlap=True, progress_cb=None):
        self.fragments = []
        total = (len(window_lengths)*len(step_sizes) if do_overlap else 0) + \
                (len(window_lengths) if do_nonoverlap else 0)
        done = 0
        for L in window_lengths:
            if do_overlap:
                for S in step_sizes:
                    cnt = 0; i = 0
                    while i + L <= self.cds_length:
                        sub = self.cds_sequence[i:i+L]
                        ok, mf = self.check_masked_overlap(i, i+L)
                        if ok:
                            ann = self.annotate(sub, f"{self.gene_name}_L{L}_S{S}_pos{i+1}",
                                                i+1, i+L, "overlapping", mf)
                            ann["window_length"] = L; ann["step_size"] = S
                            self.fragments.append(ann); cnt += 1
                        i += S
                    done += 1
                    if progress_cb: progress_cb(done/total, f"Overlapping L={L} S={S} → {cnt} frags")
            if do_nonoverlap:
                cnt = 0; i = 0
                while i + L <= self.cds_length:
                    sub = self.cds_sequence[i:i+L]
                    ok, mf = self.check_masked_overlap(i, i+L)
                    if ok:
                        ann = self.annotate(sub, f"{self.gene_name}_L{L}_NonOv_pos{i+1}",
                                            i+1, i+L, "non-overlapping", mf)
                        ann["window_length"] = L; ann["step_size"] = L
                        self.fragments.append(ann); cnt += 1
                    i += L
                done += 1
                if progress_cb: progress_cb(done/total, f"Non-overlap L={L} → {cnt} frags")
        return pd.DataFrame(self.fragments)


class dsRNAAnalyzer:
    P = ["#6C63FF","#48CAE4","#F77F00","#06D6A0","#EF476F","#FFD166",
         "#118AB2","#B5179E","#4CC9F0","#F4A261"]

    def __init__(self, df):
        self.df = df.copy(); self._derive()

    def _derive(self):
        d = self.df
        if "dinuc_composition_json" in d.columns:
            dd = d["dinuc_composition_json"].apply(lambda x: json.loads(x) if pd.notna(x) else {})
            for k in ["CG","GC","AT","TA","AA","TT"]:
                d[f"dinuc_{k}"] = dd.apply(lambda x: x.get(k,0))
        d["gc_deviation_from_50"] = abs(d["gc_percentage"] - 50)
        d["gc_in_optimal_range"]  = ((d["gc_percentage"]>=40)&(d["gc_percentage"]<=60)).astype(int)
        d["homopolymer_burden"]   = d["homopolymer_runs_6plus"]*1.0 + d["homopolymer_runs_8plus"]*3.0
        d["complexity_score"]     = 1 - d["kmer_duplication_rate"]
        d["length_category"]      = pd.cut(d["window_length"], bins=[0,250,350,600],
            labels=["Short (200-250)","Medium (300-350)","Long (400-500)"])
        cds_len = d["end_pos_1based"].max()
        d["relative_position"] = d["start_pos_1based"] / cds_len
        d["position_category"] = pd.cut(d["relative_position"], bins=[0,0.33,0.67,1.0],
            labels=["5' region","Middle","3' region"])
        d["has_design_issues"] = (d["design_issues_str"] != "none").astype(int)

    def score(self, w=None):
        d = self.df
        if w is None: w = {"gc":0.25,"cx":0.25,"hp":0.20,"ln":0.15,"mk":0.10,"ds":0.05}
        gs  = np.exp(-((d["gc_percentage"]-50)**2)/(2*10**2))
        cs  = d["complexity_score"]
        mh  = d["homopolymer_burden"].max()
        hs  = 1-(d["homopolymer_burden"]/mh) if mh>0 else 1.0
        ls  = np.exp(-((d["window_length"]-350)**2)/(2*75**2))
        ms  = 1-d["masked_overlap_fraction"]
        ds  = 1-d["has_design_issues"]
        raw = w["gc"]*gs+w["cx"]*cs+w["hp"]*hs+w["ln"]*ls+w["mk"]*ms+w["ds"]*ds
        mn, mx = raw.min(), raw.max()
        d["composite_score_normalized"] = ((raw-mn)/(mx-mn))*100
        for col, val in [("score_gc",gs),("score_complexity",cs),("score_homopolymer",hs),
                         ("score_length",ls),("score_masked",ms),("score_design",ds)]:
            d[col] = val*100

    def _ax_dark(self, ax):
        ax.set_facecolor("#161B22")
        for s in ax.spines.values(): s.set_color("#30363D")
        ax.tick_params(colors="#8B949E")
        ax.xaxis.label.set_color("#C9D1D9")
        ax.yaxis.label.set_color("#C9D1D9")
        ax.title.set_color("#E6EDF3")

    def _to_bytes(self, fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=500, bbox_inches="tight")
        buf.seek(0); data = buf.read()
        plt.close(fig); return data

    def fig_overview(self):
        d = self.df
        fig = plt.figure(figsize=(18,13), facecolor="#0D1117")
        gs  = fig.add_gridspec(3,3, hspace=0.38, wspace=0.32)

        ax = fig.add_subplot(gs[0,0]); self._ax_dark(ax)
        ax.hist(d["gc_percentage"],bins=40,color="#6C63FF",edgecolor="#0D1117",alpha=0.85)
        ax.axvline(50,color="#EF476F",lw=2,ls="--",label="50% optimal")
        ax.axvspan(40,60,alpha=0.12,color="#06D6A0",label="40-60% range")
        ax.set_xlabel("GC Content (%)"); ax.set_ylabel("Count")
        ax.set_title("A  GC Content Distribution",fontweight="bold",loc="left")
        ax.legend(fontsize=8,framealpha=0.2,labelcolor="#C9D1D9")

        ax = fig.add_subplot(gs[0,1]); self._ax_dark(ax)
        ov = [d[d["overlap_type"]=="overlapping"]["window_length"],
              d[d["overlap_type"]=="non-overlapping"]["window_length"]]
        ax.hist(ov,bins=5,label=["Overlapping","Non-overlapping"],
                color=["#F77F00","#48CAE4"],edgecolor="#0D1117",alpha=0.85)
        ax.set_xlabel("Fragment Length (bp)"); ax.set_ylabel("Count")
        ax.set_title("B  Fragment Length by Design",fontweight="bold",loc="left")
        ax.legend(fontsize=8,framealpha=0.2,labelcolor="#C9D1D9")

        ax = fig.add_subplot(gs[0,2]); self._ax_dark(ax)
        ax.hist(d["complexity_score"],bins=40,color="#06D6A0",edgecolor="#0D1117",alpha=0.85)
        med = d["complexity_score"].median()
        ax.axvline(med,color="#FFD166",lw=2,ls="--",label=f"Median {med:.3f}")
        ax.set_xlabel("Complexity Score"); ax.set_ylabel("Count")
        ax.set_title("C  Sequence Complexity",fontweight="bold",loc="left")
        ax.legend(fontsize=8,framealpha=0.2,labelcolor="#C9D1D9")

        ax = fig.add_subplot(gs[1,0]); self._ax_dark(ax)
        bins = np.arange(0,d["homopolymer_runs_6plus"].max()+2,1)
        ax.hist(d["homopolymer_runs_6plus"],bins=bins,color="#EF476F",edgecolor="#0D1117",alpha=0.85)
        ax.set_xlabel("Homopolymer Runs (>=6bp)"); ax.set_ylabel("Count")
        ax.set_title("D  Homopolymer Burden",fontweight="bold",loc="left")

        ax = fig.add_subplot(gs[1,1]); self._ax_dark(ax)
        iss  = d["design_issues_str"].value_counts()
        cols = [self.P[i%len(self.P)] for i in range(len(iss))]
        ax.barh(range(len(iss)),iss.values,color=cols,edgecolor="#0D1117",height=0.65)
        ax.set_yticks(range(len(iss)))
        ax.set_yticklabels([l.replace("_"," ").title() for l in iss.index],fontsize=8,color="#C9D1D9")
        ax.set_xlabel("Fragments"); ax.set_title("E  Design Quality Issues",fontweight="bold",loc="left")

        ax = fig.add_subplot(gs[1,2]); ax.set_facecolor("#161B22")
        ax.set_title("F  Distribution Along CDS",fontweight="bold",loc="left",color="#E6EDF3")
        pc = d["position_category"].value_counts()
        _, _, autos = ax.pie(pc.values,labels=pc.index,autopct="%1.1f%%",
            colors=["#6C63FF","#48CAE4","#06D6A0"],startangle=90,
            textprops={"color":"#C9D1D9","fontsize":9},
            wedgeprops={"edgecolor":"#0D1117","linewidth":1.5})
        for a in autos: a.set_color("#E6EDF3")

        ax = fig.add_subplot(gs[2,0]); self._ax_dark(ax)
        sc = ax.scatter(d["gc_percentage"],d["complexity_score"],
            c=d["composite_score_normalized"],cmap="plasma",s=25,alpha=0.65,edgecolors="none")
        ax.axvline(50,color="#30363D",lw=1,ls="--")
        ax.set_xlabel("GC Content (%)"); ax.set_ylabel("Complexity Score")
        ax.set_title("G  GC vs Complexity",fontweight="bold",loc="left")
        cb = plt.colorbar(sc,ax=ax); cb.set_label("Composite Score",color="#8B949E")
        plt.setp(cb.ax.yaxis.get_ticklabels(),color="#8B949E")

        ax = fig.add_subplot(gs[2,1]); self._ax_dark(ax)
        lcats = d["length_category"].dropna().unique()
        bdata = [d[d["length_category"]==c]["composite_score_normalized"].values for c in lcats]
        bp = ax.boxplot(bdata,labels=lcats,patch_artist=True,
            medianprops=dict(color="#FFD166",linewidth=2.5),
            whiskerprops=dict(color="#8B949E"),capprops=dict(color="#8B949E"),
            flierprops=dict(marker="o",markerfacecolor="#EF476F",markersize=3,alpha=0.5))
        for i,box in enumerate(bp["boxes"]):
            box.set_facecolor(self.P[i%len(self.P)]+"44"); box.set_edgecolor(self.P[i%len(self.P)])
        ax.set_ylabel("Composite Score"); ax.set_xlabel("Length Category")
        ax.set_title("H  Score by Length",fontweight="bold",loc="left"); ax.tick_params(axis="x",rotation=12)

        ax = fig.add_subplot(gs[2,2]); self._ax_dark(ax)
        ax.hist(d["composite_score_normalized"],bins=40,color="#B5179E",edgecolor="#0D1117",alpha=0.85)
        med2 = d["composite_score_normalized"].median()
        ax.axvline(med2,color="#FFD166",lw=2,ls="--",label=f"Median {med2:.1f}")
        ax.axvline(70,color="#06D6A0",lw=2,ls="--",label="Threshold 70")
        ax.set_xlabel("Composite Score"); ax.set_ylabel("Count")
        ax.set_title("I  Composite Score",fontweight="bold",loc="left")
        ax.legend(fontsize=8,framealpha=0.2,labelcolor="#C9D1D9")

        fig.suptitle("dsRNA Fragment Analysis — Overview",fontsize=16,fontweight="bold",color="#E6EDF3",y=0.998)
        return self._to_bytes(fig)

    def fig_radar(self, top_n=10):
        top   = self.df.nlargest(top_n,"composite_score_normalized")
        cats  = ["GC","Complexity","Homopolymer","Length","Masked","Design"]
        cols  = ["score_gc","score_complexity","score_homopolymer","score_length","score_masked","score_design"]
        N     = len(cats)
        angles= [n/float(N)*2*np.pi for n in range(N)]+[0]
        fig, axes = plt.subplots(2,5,figsize=(22,9),subplot_kw=dict(polar=True),facecolor="#0D1117")
        axes = axes.flatten()
        clrs = plt.cm.plasma(np.linspace(0.15,0.85,top_n))
        for idx,(_,row) in enumerate(top.iterrows()):
            if idx >= len(axes): break
            vals = [row[c] for c in cols]+[row[cols[0]]]
            ax = axes[idx]; ax.set_facecolor("#161B22")
            ax.plot(angles,vals,"o-",lw=2,color=clrs[idx])
            ax.fill(angles,vals,alpha=0.2,color=clrs[idx])
            ax.set_xticks(angles[:-1]); ax.set_xticklabels(cats,size=8,color="#C9D1D9")
            ax.set_ylim(0,100); ax.tick_params(colors="#8B949E")
            ax.set_title(f"#{idx+1}  {row['fragment_id'][:18]}\nScore {row['composite_score_normalized']:.1f}",
                         size=8,color="#E6EDF3",pad=10)
            ax.grid(True,alpha=0.25,color="#30363D")
            for sp in ax.spines.values(): sp.set_color("#30363D")
        fig.suptitle(f"Top {top_n} Candidate Score Components",fontsize=14,fontweight="bold",color="#E6EDF3")
        plt.tight_layout()
        return self._to_bytes(fig)

    def fig_heatmap(self):
        d = self.df; cds_len = d["end_pos_1based"].max()
        bins = 100; bin_sz = cds_len/bins
        lengths = sorted(d["window_length"].unique())
        matrix  = np.zeros((len(lengths),bins))
        for li,L in enumerate(lengths):
            for _,row in d[d["window_length"]==L].iterrows():
                s = int((row["start_pos_1based"]-1)/bin_sz)
                e = int(row["end_pos_1based"]/bin_sz)
                matrix[li,s:min(e,bins)] += row["composite_score_normalized"]
        fig,ax = plt.subplots(figsize=(16,6),facecolor="#0D1117")
        ax.set_facecolor("#161B22")
        im = ax.imshow(matrix,aspect="auto",cmap="plasma",extent=[0,cds_len,0,len(lengths)])
        ax.set_yticks(range(len(lengths))); ax.set_yticklabels([f"{L} bp" for L in lengths],color="#C9D1D9")
        ax.set_xlabel("CDS Position (bp)",color="#C9D1D9"); ax.set_ylabel("Fragment Length",color="#C9D1D9")
        ax.set_title("Genomic Coverage Heatmap (Score Intensity)",fontweight="bold",color="#E6EDF3",pad=12)
        ax.tick_params(colors="#8B949E")
        cb = plt.colorbar(im,ax=ax); cb.set_label("Cumulative Score",color="#8B949E")
        plt.setp(cb.ax.yaxis.get_ticklabels(),color="#8B949E")
        plt.tight_layout(); return self._to_bytes(fig)

    def fig_pca(self):
        feats   = ["gc_percentage","complexity_score","homopolymer_burden",
                   "window_length","masked_overlap_fraction","relative_position"]
        df_feat = self.df[feats].dropna()
        X       = StandardScaler().fit_transform(df_feat)
        pca     = PCA(n_components=3); pcs = pca.fit_transform(X)
        ev      = pca.explained_variance_ratio_
        scores  = self.df.loc[df_feat.index,"composite_score_normalized"]
        fig,axes= plt.subplots(1,3,figsize=(18,5),facecolor="#0D1117")
        for ax in axes: ax.set_facecolor("#161B22")
        sc = axes[0].scatter(pcs[:,0],pcs[:,1],c=scores,cmap="plasma",s=20,alpha=0.65,edgecolors="none")
        axes[0].set_xlabel(f"PC1 ({ev[0]*100:.1f}%)",color="#C9D1D9")
        axes[0].set_ylabel(f"PC2 ({ev[1]*100:.1f}%)",color="#C9D1D9")
        axes[0].set_title("A  PC1 vs PC2",fontweight="bold",color="#E6EDF3")
        cb = plt.colorbar(sc,ax=axes[0]); cb.set_label("Composite Score",color="#8B949E")
        plt.setp(cb.ax.yaxis.get_ticklabels(),color="#8B949E")
        axes[1].bar([1,2,3],ev*100,color=["#6C63FF","#48CAE4","#06D6A0"],edgecolor="#0D1117")
        axes[1].set_xlabel("PC",color="#C9D1D9"); axes[1].set_ylabel("Variance (%)",color="#C9D1D9")
        axes[1].set_title("B  Scree Plot",fontweight="bold",color="#E6EDF3"); axes[1].set_xticks([1,2,3])
        ld = pca.components_.T
        axes[2].barh(feats,ld[:,0],color=self.P[:len(feats)],edgecolor="#0D1117")
        axes[2].set_xlabel("PC1 Loading",color="#C9D1D9")
        axes[2].set_title("C  PC1 Feature Loadings",fontweight="bold",color="#E6EDF3")
        axes[2].axvline(0,color="#8B949E",lw=1)
        for l in axes[2].get_yticklabels(): l.set_color("#C9D1D9")
        for ax in axes:
            ax.tick_params(colors="#8B949E")
            for sp in ax.spines.values(): sp.set_color("#30363D")
        fig.suptitle("PCA — dsRNA Fragment Space",fontsize=15,fontweight="bold",color="#E6EDF3")
        plt.tight_layout(); return self._to_bytes(fig)

    def fig_correlation(self):
        fm  = {"GC %":"gc_percentage","Complexity":"complexity_score","Homopolymer":"homopolymer_burden",
               "Length":"window_length","Masked Ov.":"masked_overlap_fraction",
               "CDS Pos.":"relative_position","Score":"composite_score_normalized"}
        sub = self.df[[v for v in fm.values()]].copy(); sub.columns = list(fm.keys())
        corr = sub.corr(); mask = np.triu(np.ones_like(corr,dtype=bool),k=1)
        fig,ax = plt.subplots(figsize=(10,8),facecolor="#0D1117"); ax.set_facecolor("#161B22")
        sns.heatmap(corr,mask=mask,annot=True,fmt=".2f",cmap="coolwarm",center=0,square=True,
                    linewidths=0.8,linecolor="#0D1117",cbar_kws={"shrink":0.78},ax=ax,vmin=-1,vmax=1,
                    annot_kws={"size":10,"color":"#E6EDF3"})
        ax.set_title("Feature Correlation Matrix",fontweight="bold",fontsize=14,color="#E6EDF3",pad=16)
        ax.tick_params(colors="#C9D1D9")
        plt.tight_layout(); return self._to_bytes(fig)

    def fig_top_candidates(self, top_n=10):
        top  = self.df.nlargest(top_n,"composite_score_normalized")
        fig, axes = plt.subplots(2,2,figsize=(18,11),facecolor="#0D1117")
        for ax in axes.flatten():
            ax.set_facecolor("#161B22")
            for s in ax.spines.values(): s.set_color("#30363D")
            ax.tick_params(colors="#8B949E")
        y = np.arange(len(top)); cmap = plt.cm.plasma
        clrs = [cmap(v/100) for v in top["composite_score_normalized"].values]
        bars = axes[0,0].barh(y,top["composite_score_normalized"],color=clrs,edgecolor="#0D1117",height=0.7)
        axes[0,0].set_yticks(y)
        axes[0,0].set_yticklabels([r[:28] for r in top["fragment_id"]],fontsize=8,color="#C9D1D9")
        axes[0,0].set_xlabel("Composite Score",color="#C9D1D9")
        axes[0,0].set_title("A  Top Candidates by Score",fontweight="bold",color="#E6EDF3",loc="left")
        for bar,sc in zip(bars,top["composite_score_normalized"]):
            axes[0,0].text(sc+0.5,bar.get_y()+bar.get_height()/2,f"{sc:.1f}",
                           va="center",fontsize=8,color="#FFD166",fontweight="bold")
        fcols = ["score_gc","score_complexity","score_homopolymer","score_length","score_masked","score_design"]
        flabs = ["GC","Complexity","Homopoly","Length","Masked","Design"]
        im = axes[0,1].imshow(top[fcols].values,aspect="auto",cmap="plasma",vmin=0,vmax=100)
        axes[0,1].set_xticks(range(len(flabs))); axes[0,1].set_xticklabels(flabs,rotation=35,ha="right",color="#C9D1D9")
        axes[0,1].set_yticks(range(len(top)))
        axes[0,1].set_yticklabels([r[:22] for r in top["fragment_id"]],fontsize=8,color="#C9D1D9")
        axes[0,1].set_title("B  Score Component Breakdown",fontweight="bold",color="#E6EDF3",loc="left")
        cb = plt.colorbar(im,ax=axes[0,1]); cb.set_label("Score",color="#8B949E")
        plt.setp(cb.ax.yaxis.get_ticklabels(),color="#8B949E")
        x = np.arange(len(top)); w = 0.35
        axes[1,0].bar(x-w/2,top["gc_percentage"],w,color="#48CAE4",edgecolor="#0D1117",alpha=0.9)
        ax2 = axes[1,0].twinx(); ax2.set_facecolor("#161B22")
        ax2.bar(x+w/2,top["complexity_score"]*100,w,color="#F77F00",edgecolor="#0D1117",alpha=0.9)
        axes[1,0].set_xlabel("Rank",color="#C9D1D9"); axes[1,0].set_ylabel("GC Content (%)",color="#48CAE4")
        ax2.set_ylabel("Complexity Score",color="#F77F00")
        axes[1,0].set_xticks(x); axes[1,0].set_xticklabels([f"#{i+1}" for i in range(len(top))],color="#C9D1D9")
        axes[1,0].axhline(50,color="#30363D",ls="--",lw=1)
        axes[1,0].tick_params(axis="y",colors="#48CAE4"); ax2.tick_params(colors="#F77F00")
        for s in ax2.spines.values(): s.set_color("#30363D")
        axes[1,0].set_title("C  GC vs Complexity",fontweight="bold",color="#E6EDF3",loc="left")
        cds_len = self.df["end_pos_1based"].max()
        for i,(_,row) in enumerate(top.iterrows()):
            col = cmap(row["composite_score_normalized"]/100)
            axes[1,1].barh(i,row["end_pos_1based"]-row["start_pos_1based"],
                           left=row["start_pos_1based"],height=0.7,color=col,edgecolor="#0D1117")
        axes[1,1].set_yticks(range(len(top)))
        axes[1,1].set_yticklabels([f"#{i+1}" for i in range(len(top))],color="#C9D1D9")
        axes[1,1].set_xlabel("CDS Position (bp)",color="#C9D1D9"); axes[1,1].set_xlim(0,cds_len)
        axes[1,1].set_title("D  Fragment Positions on CDS",fontweight="bold",color="#E6EDF3",loc="left")
        fig.suptitle(f"Top {top_n} dsRNA Candidates",fontsize=16,fontweight="bold",color="#E6EDF3")
        plt.tight_layout(); return self._to_bytes(fig)

    def build_report(self, top_n):
        d = self.df; buf = io.StringIO()
        buf.write("="*80+"\ndsRNA FRAGMENT ANALYSIS — REPORT\n"+"="*80+"\n\n")
        buf.write(f"Total fragments : {len(d)}\nCDS length      : {d['end_pos_1based'].max()} bp\n")
        buf.write(f"Overlapping     : {len(d[d['overlap_type']=='overlapping'])}\n")
        buf.write(f"Non-overlapping : {len(d[d['overlap_type']=='non-overlapping'])}\n\n")
        buf.write(f"GC mean         : {d['gc_percentage'].mean():.2f}%\n")
        buf.write(f"Score mean      : {d['composite_score_normalized'].mean():.2f}\n\n")
        buf.write(f"TOP {top_n} CANDIDATES\n"+"-"*60+"\n\n")
        for rank,(_,row) in enumerate(d.nlargest(top_n,"composite_score_normalized").iterrows(),1):
            buf.write(f"#{rank:02d}  {row['fragment_id']}\n"
                      f"    Pos {row['start_pos_1based']}-{row['end_pos_1based']} bp  "
                      f"Len {row['window_length']} bp  GC {row['gc_percentage']:.1f}%  "
                      f"Score {row['composite_score_normalized']:.2f}  Issues: {row['design_issues_str']}\n\n")
        return buf.getvalue()

    def build_csv(self, top_n):
        cols = ["fragment_id","gene_name","start_pos_1based","end_pos_1based","window_length",
                "overlap_type","composite_score_normalized","score_gc","score_complexity",
                "score_homopolymer","score_length","score_masked","score_design","gc_percentage",
                "complexity_score","homopolymer_runs_6plus","homopolymer_runs_8plus",
                "kmer_duplication_rate","masked_overlap_percentage","design_issues_str","sequence"]
        ranked = self.df.sort_values("composite_score_normalized",ascending=False)[cols]
        return ranked.to_csv(index=False).encode(), ranked.head(top_n).to_csv(index=False).encode()


# =====================================================================
#  STREAMLIT UI
# =====================================================================

st.set_page_config(
    page_title="dsRNA Design Suite",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"]{background:#0D1117;color:#E6EDF3;}
[data-testid="stSidebar"]{background:#161B22;border-right:1px solid #21262D;}
.stButton>button{background:#6C63FF;color:#fff;border:none;border-radius:8px;
  font-weight:700;font-size:16px;padding:12px 32px;width:100%;transition:background .2s;}
.stButton>button:hover{background:#8B86FF;}
.stDownloadButton>button{background:#06D6A0;color:#0D1117;border:none;border-radius:6px;
  font-weight:600;padding:6px 18px;}
.stDownloadButton>button:hover{background:#05b889;}
h1,h2,h3{color:#E6EDF3 !important;}
.mcard{background:#161B22;border:1px solid #21262D;border-radius:10px;
  padding:16px 20px;text-align:center;}
.mval{font-size:28px;font-weight:800;color:#6C63FF;}
.mlbl{font-size:12px;color:#8B949E;margin-top:2px;}
.ftitle{color:#48CAE4;font-weight:700;font-size:15px;margin:18px 0 6px;}
.badge{display:inline-block;padding:3px 10px;border-radius:12px;font-size:11px;
  font-weight:700;margin:2px;}
</style>
""", unsafe_allow_html=True)

# ── Header ──────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(90deg,#6C63FF,#48CAE4);
            padding:22px 28px;border-radius:12px;margin-bottom:20px;">
  <div style="font-size:28px;font-weight:900;color:#fff;letter-spacing:1px;">
    🧬 dsRNA Design Suite
  </div>
  <div style="color:#ffffffbb;font-size:13px;margin-top:4px;">
    S Nanda &nbsp;·&nbsp; C Bose &nbsp;·&nbsp; A Nayak
    &nbsp;&nbsp;|&nbsp;&nbsp;
    FASTA → Sliding-window fragments → Score &amp; Rank → 500-DPI PNG Figures
  </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Pipeline Parameters")
    st.markdown("---")
    st.markdown("**Fragment Generation**")
    lengths_str   = st.text_input("Window lengths (bp)", value="200 250 300 400 500")
    steps_str     = st.text_input("Step sizes — overlapping", value="25 50")
    tolerance     = st.slider("Mask tolerance", 0.0, 0.5, 0.10, 0.01)
    do_overlap    = st.checkbox("Overlapping fragments",     value=True)
    do_nonoverlap = st.checkbox("Non-overlapping fragments", value=True)
    st.markdown("---")
    st.markdown("**Scoring & Report**")
    top_n = st.slider("Top N candidates", 5, 50, 20)
    st.markdown("---")
    st.markdown("**Scoring Weights**")
    with st.expander("Adjust weights (advanced)"):
        w_gc = st.slider("GC content",   0.0,1.0,0.25,0.05)
        w_cx = st.slider("Complexity",   0.0,1.0,0.25,0.05)
        w_hp = st.slider("Homopolymer",  0.0,1.0,0.20,0.05)
        w_ln = st.slider("Length",       0.0,1.0,0.15,0.05)
        w_mk = st.slider("Masked",       0.0,1.0,0.10,0.05)
        w_ds = st.slider("Design clean", 0.0,1.0,0.05,0.05)
        tot  = w_gc+w_cx+w_hp+w_ln+w_mk+w_ds
        if abs(tot-1.0) > 0.01: st.warning(f"Weights sum to {tot:.2f} (ideally 1.00)")
    weights = {"gc":w_gc,"cx":w_cx,"hp":w_hp,"ln":w_ln,"mk":w_mk,"ds":w_ds}
    st.markdown("---")
    st.caption("Free hosting on Streamlit Cloud\nUpload FASTA → Run Pipeline → Download results")

# ── File uploads ─────────────────────────────────────────────────────
col_a, col_b = st.columns([2,1])
with col_a:
    st.markdown("#### 📂 Input Files")
    fasta_file = st.file_uploader("CDS sequence — FASTA *(required)*",
                                  type=["fasta","fa","fna","txt"])
    mask_file  = st.file_uploader("Masked regions — tab-delimited *(optional)*",
                                  type=["txt","tsv"])
    gene_name  = st.text_input("Gene name override (optional)")
with col_b:
    st.markdown("#### ℹ️ File formats")
    st.code(">gene_id\nATGCGTAACCGT...", language="text")
    st.caption("Masked regions: tab-separated start\\tend (1-based, one per line)")

st.markdown("---")
run_clicked = st.button("🚀  RUN FULL PIPELINE", use_container_width=True)

# ── Pipeline ─────────────────────────────────────────────────────────
if run_clicked:
    if fasta_file is None:
        st.error("Please upload a FASTA file first."); st.stop()
    try:
        window_lengths = [int(x) for x in lengths_str.split()]
        step_sizes     = [int(x) for x in steps_str.split()]
    except ValueError:
        st.error("Window lengths and step sizes must be space-separated integers."); st.stop()

    with st.spinner("Reading FASTA..."):
        record = SeqIO.read(io.StringIO(fasta_file.read().decode("utf-8")), "fasta")
        gene   = gene_name.strip() or record.id
        cds    = str(record.seq)
        st.info(f"Gene: **{gene}**   CDS: **{len(cds)} bp**")

    masked = []
    if mask_file:
        for line in mask_file.read().decode("utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                parts = line.split("\t")
                if len(parts) >= 2: masked.append((int(parts[0]),int(parts[1])))
        st.info(f"Loaded **{len(masked)}** masked regions")

    pbar = st.progress(0.0, text="Starting...")

    def upd(frac, msg): pbar.progress(min(frac*0.45, 0.45), text=msg)

    with st.spinner("Generating fragments..."):
        designer = dsRNADesigner(cds, gene, masked, tolerance)
        df = designer.generate_fragments(window_lengths, step_sizes,
                                         do_overlap, do_nonoverlap, upd)

    if len(df) == 0:
        st.error("No fragments generated. Check FASTA and parameters."); st.stop()

    pbar.progress(0.50, text="Scoring...")
    analyzer = dsRNAAnalyzer(df)
    analyzer.score(weights)
    pbar.progress(0.55, text="Rendering figures...")

    figs_spec = [
        ("Figure 1 — Overview Panel",       "Figure1_Overview.png",        analyzer.fig_overview),
        ("Figure 2 — Radar Score Charts",    "Figure2_RadarCharts.png",     lambda: analyzer.fig_radar(top_n=10)),
        ("Figure 3 — Coverage Heatmap",      "Figure3_CoverageHeatmap.png", analyzer.fig_heatmap),
        ("Figure 4 — PCA Analysis",          "Figure4_PCA.png",             analyzer.fig_pca),
        ("Figure 5 — Correlation Heatmap",   "Figure5_Correlation.png",     analyzer.fig_correlation),
        ("Figure 6 — Top Candidates",        "Figure6_TopCandidates.png",   lambda: analyzer.fig_top_candidates(top_n=10)),
    ]
    fig_bytes = {}
    for i,(label,fname,fn) in enumerate(figs_spec):
        pbar.progress(0.55+i*0.07, text=f"Rendering {label}..."); fig_bytes[fname] = fn()

    pbar.progress(0.98, text="Building report...")
    report_txt     = analyzer.build_report(top_n)
    all_csv,top_csv= analyzer.build_csv(top_n)
    tsv_bytes      = df.to_csv(sep="\t",index=False).encode()
    pbar.progress(1.0, text="Done!")

    # ── Results ──────────────────────────────────────────────────────
    st.success(f"Pipeline complete — **{len(df)}** fragments generated and scored.")
    st.markdown("---")
    st.markdown("### 📊 Summary")
    d = analyzer.df
    c1,c2,c3,c4,c5 = st.columns(5)
    for col,val,lbl in [
        (c1, len(d),                                       "Total Fragments"),
        (c2, len(d[d["overlap_type"]=="overlapping"]),      "Overlapping"),
        (c3, len(d[d["overlap_type"]=="non-overlapping"]),  "Non-overlapping"),
        (c4, f"{d['composite_score_normalized'].mean():.1f}","Mean Score"),
        (c5, len(d[d["composite_score_normalized"]>=70]),   "Score ≥ 70"),
    ]:
        col.markdown(f'<div class="mcard"><div class="mval">{val}</div>'
                     f'<div class="mlbl">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("### 🏆 Top Candidates")
    top_df = d.nlargest(top_n,"composite_score_normalized")[
        ["fragment_id","start_pos_1based","end_pos_1based","window_length",
         "overlap_type","gc_percentage","complexity_score",
         "composite_score_normalized","design_issues_str"]].reset_index(drop=True)
    top_df.index += 1
    top_df.columns = ["Fragment ID","Start","End","Length","Type","GC%","Complexity","Score","Issues"]
    st.dataframe(top_df.style.background_gradient(subset=["Score"],cmap="plasma"),
                 use_container_width=True)

    st.markdown("### 📥 Downloads")
    dc1,dc2,dc3,dc4 = st.columns(4)
    dc1.download_button("⬇ All Fragments TSV",     tsv_bytes,  "fragments.tsv","text/tab-separated-values")
    dc2.download_button("⬇ Ranked Candidates CSV", all_csv,    "ranked_candidates.csv","text/csv")
    dc3.download_button(f"⬇ Top {top_n} CSV",      top_csv,   f"top{top_n}_candidates.csv","text/csv")
    dc4.download_button("⬇ Analysis Report TXT",   report_txt.encode(),"dsRNA_report.txt","text/plain")

    st.markdown("### 🖼️ Figures  *(500 DPI PNG)*")
    for label,fname,_ in figs_spec:
        st.markdown(f'<div class="ftitle">🔬 {label}</div>', unsafe_allow_html=True)
        st.image(fig_bytes[fname], use_container_width=True)
        st.download_button(f"⬇ {fname}", fig_bytes[fname], fname, "image/png", key=fname)

else:
    st.markdown("""
<div style="background:#161B22;border:1px solid #21262D;border-radius:12px;
            padding:36px;text-align:center;margin-top:16px;">
  <div style="font-size:48px;">🧬</div>
  <div style="font-size:20px;font-weight:700;color:#E6EDF3;margin:12px 0 8px;">
    Upload your FASTA and click Run Pipeline
  </div>
  <div style="color:#8B949E;font-size:14px;max-width:560px;margin:auto;line-height:1.6;">
    Generates sliding-window dsRNA fragment candidates, scores them on GC content,
    sequence complexity, homopolymer burden, length optimality and masked-region overlap,
    then produces 6 publication-quality 500-DPI PNG figures — all in one click.
  </div>
  <div style="margin-top:20px;">
    <span class="badge" style="background:#6C63FF22;color:#6C63FF;">FASTA Input</span>
    <span class="badge" style="background:#48CAE422;color:#48CAE4;">Sliding Window</span>
    <span class="badge" style="background:#06D6A022;color:#06D6A0;">Composite Scoring</span>
    <span class="badge" style="background:#F77F0022;color:#F77F00;">500 DPI PNG</span>
    <span class="badge" style="background:#EF476F22;color:#EF476F;">No PDF</span>
    <span class="badge" style="background:#FFD16622;color:#FFD166;">Free &amp; Public</span>
  </div>
</div>
""", unsafe_allow_html=True)
