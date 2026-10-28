"""YOLO/ByteTrack analyzer for fixed-camera futsal footage."""
from __future__ import annotations
import argparse, csv, json, logging, math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np

LOG = logging.getLogger("futsal")
DEFAULT_VIDEO = Path(r"D:\Football Match Analyzer\Manzano-Jesolo.App.mp4")
PLAYERS = {"person", "player", "goalkeeper", "goalie", "keeper", "football player",
           "soccer player", "futsal player"}
BALLS = {"sports ball", "ball", "football", "soccer ball", "futsal ball"}
OFFICIALS = {"referee", "official", "umpire", "coach"}


def args():
    p = argparse.ArgumentParser(description="Detect, track, and summarize a futsal video.")
    p.add_argument("video", nargs="?", type=Path, default=DEFAULT_VIDEO)
    p.add_argument("--model", default="yolo11n.pt", help="COCO or custom football detector weights")
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--stride", type=int, default=3, help="Analyze every Nth frame")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--confidence", type=float, default=.18)
    p.add_argument("--possession-radius", type=float, default=100, help="Distance in source pixels")
    p.add_argument("--device", help="Optional Ultralytics device, e.g. 0 or cpu")
    p.add_argument("--team1-name", default="Team 1")
    p.add_argument("--team2-name", default="Team 2")
    p.add_argument("--no-render", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def roles_for(names):
    items = names.items() if isinstance(names, dict) else enumerate(names)
    roles = {}
    for cid, name in items:
        label = " ".join(str(name).lower().replace("_", " ").split())
        if label in BALLS: roles[int(cid)] = "ball"
        elif label in OFFICIALS: roles[int(cid)] = "official"
        elif label in PLAYERS: roles[int(cid)] = "player"
    if not roles:
        raise ValueError("Model has no recognized player/ball class names; use COCO or custom player/ball weights.")
    if "ball" not in roles.values(): LOG.warning("No ball class; possession will be unavailable.")
    return list(roles), roles


def shirt_color(frame, box):
    x1,y1,x2,y2 = box; w,h=x2-x1,y2-y1
    if w < 8 or h < 16: return None
    crop=frame[y1+int(h*.18):y1+int(h*.52), x1+int(w*.25):x1+int(w*.75)]
    if crop.size == 0: return None
    lab=cv2.cvtColor(crop,cv2.COLOR_BGR2LAB).reshape(-1,3)
    return np.median(lab,axis=0).tolist()


def cluster(features):
    ids=[]; vectors=[]
    for tid,samples in features.items():
        if len(samples)>=3:
            ids.append(tid); vectors.append(np.median(np.asarray(samples,dtype=np.float32),axis=0))
    if len(vectors)<2: return {}, {"status":"insufficient player tracks","clusters":[]}
    crit=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,80,.2)
    _,labels,centers=cv2.kmeans(np.asarray(vectors,dtype=np.float32),2,None,crit,8,cv2.KMEANS_PP_CENTERS)
    mapping={tid:int(label) for tid,label in zip(ids,labels.ravel())}
    colors=[]
    for i,c in enumerate(centers):
        bgr=cv2.cvtColor(np.uint8(np.clip(c,0,255)).reshape(1,1,3),cv2.COLOR_LAB2BGR)[0,0]
        colors.append({"cluster":i,"approx_jersey_rgb":[int(x) for x in bgr[::-1]],
                       "track_count":sum(v==i for v in mapping.values())})
    return mapping,{"status":"estimated","method":"per-track upper-body Lab color, k-means k=2",
                    "clusters":colors,"warning":"Cluster order is arbitrary; verify which color is Manzano."}


def summarize(rows,fps,stride,radius,team_ids,names,color_info):
    step=stride/fps; frames=defaultdict(list); positions=defaultdict(list); seen=Counter()
    for r in rows:
        frames[r["frame_index"]].append(r)
        if r["object_type"]=="player" and r["track_id"] is not None:
            seen[r["track_id"]]+=1
            positions[r["track_id"]].append((r["frame_index"],r["center_x"],r["center_y"]))
    control=Counter(); ball_frames=0
    for group in frames.values():
        ps=[r for r in group if r["object_type"]=="player" and r["track_id"] is not None]
        bs=[r for r in group if r["object_type"]=="ball"]
        if not ps or not bs: continue
        ball_frames+=1; b=max(bs,key=lambda r:r["confidence"])
        nearest=min(ps,key=lambda r:math.hypot(r["center_x"]-b["center_x"],r["center_y"]-b["center_y"]))
        d=math.hypot(nearest["center_x"]-b["center_x"],nearest["center_y"]-b["center_y"])
        team=team_ids.get(nearest["track_id"])
        if d<=radius and team is not None: control[team]+=1
    total=sum(control.values())
    teams=[{"team":n,"estimated_possession_seconds":round(control[i]*step,2),
            "estimated_possession_share_percent":round(control[i]*100/total,1) if total else None,
            "player_tracks":sum(v==i for v in team_ids.values())} for i,n in enumerate(names)]
    players=[]
    for tid,count in seen.items():
        pts=sorted(positions[tid])
        dist=sum(math.hypot(b[1]-a[1],b[2]-a[2]) for a,b in zip(pts,pts[1:]))
        players.append({"track_id":tid,"team":names[team_ids[tid]] if tid in team_ids else None,
                        "observed_seconds":round(count*step,2),"observed_frames":count,"distance_px":round(dist,1)})
    players.sort(key=lambda p:p["observed_seconds"],reverse=True)
    return {"method":"YOLO + ByteTrack + jersey-color clusters + nearest-player ball heuristic",
      "sampling_stride":stride,"analyzed_fps":round(fps/stride,3),"team_color_clustering":color_info,
      "teams":teams,"players":players,"ball_observed_sample_frames":ball_frames,
      "possession_assigned_sample_frames":total,"limitations":[
        "Possession uses proximity, not verified ball contact.","Small ball detections may be missed; tracks may fragment.",
        "Verify jersey clusters against team kits before interpreting team statistics.",
        "Movement is in image pixels; court calibration is required for meters.",
        "Passes, shots, goals, fouls, and tactical actions are not classified."]}


def write_csv(path,rows,team_ids,names):
    fields=["timestamp_s","frame_index","object_type","class_name","confidence","track_id","team",
            "x1","y1","x2","y2","center_x","center_y"]
    with path.open("w",newline="",encoding="utf-8-sig") as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore"); writer.writeheader()
        for original in rows:
            r=dict(original); tid=r.get("track_id")
            r["team"]=names[team_ids[tid]] if tid in team_ids else ""
            writer.writerow(r)


def render(video,dest,fps,stride,by_frame,team_ids,names):
    cap=cv2.VideoCapture(str(video)); w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out=cv2.VideoWriter(str(dest),cv2.VideoWriter_fourcc(*"mp4v"),fps/stride,(w,h))
    if not out.isOpened(): cap.release(); raise RuntimeError(f"Cannot create annotated video {dest}")
    palette=[(70,210,255),(255,130,70)]; index=0
    try:
        while True:
            ok,frame=cap.read()
            if not ok: break
            if index%stride==0:
                for r in by_frame.get(index,[]):
                    x1,y1,x2,y2=(int(r[k]) for k in ("x1","y1","x2","y2")); tid=r["track_id"]
                    team=team_ids.get(tid) if tid is not None else None; ball=r["object_type"]=="ball"
                    color=(40,240,255) if ball else (150,150,150) if team is None else palette[team]
                    label=f"Ball {r['confidence']:.2f}" if ball else f"{names[team]} #{tid}" if team is not None else f"Unassigned #{tid}"
                    cv2.rectangle(frame,(x1,y1),(x2,y2),color,2 if ball else 3); y=max(22,y1-7)
                    cv2.rectangle(frame,(x1,y-22),(min(w-1,x1+245),y+3),color,-1)
                    cv2.putText(frame,label,(x1+4,y-4),cv2.FONT_HERSHEY_SIMPLEX,.58,(10,10,10),2,cv2.LINE_AA)
                out.write(frame)
            index+=1
    finally: cap.release(); out.release()


def run(a):
    video=a.video.expanduser().resolve()
    if not video.is_file(): raise FileNotFoundError(f"Video not found: {video}")
    if a.stride<1: raise ValueError("--stride must be at least 1")
    out=a.output_dir or Path(__file__).resolve().parents[1]/"outputs"/f"{video.stem}_{datetime.now():%Y%m%d_%H%M%S}"
    out=out.expanduser().resolve(); out.mkdir(parents=True,exist_ok=True)
    try: from ultralytics import YOLO
    except ImportError as e: raise RuntimeError("Install dependencies: python -m pip install -r requirements.txt") from e
    LOG.info("Loading %s",a.model); model=YOLO(a.model); classes,role_map=roles_for(model.names)
    cap=cv2.VideoCapture(str(video))
    if not cap.isOpened(): raise RuntimeError(f"OpenCV cannot open {video}")
    fps=float(cap.get(cv2.CAP_PROP_FPS)); count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not math.isfinite(fps) or fps<=0: cap.release(); raise ValueError("Video frame rate is invalid.")
    rows=[]; by_frame=defaultdict(list); features=defaultdict(list); index=sampled=0
    try:
        while True:
            ok,frame=cap.read()
            if not ok: break
            if index%a.stride==0:
                opts={"persist":True,"tracker":"bytetrack.yaml","classes":classes,
                      "conf":a.confidence,"imgsz":a.imgsz,"verbose":False}
                if a.device: opts["device"]=a.device
                result=model.track(frame,**opts)[0]; current=[]; boxes=result.boxes
                if boxes is not None and len(boxes):
                    coords=boxes.xyxy.cpu().numpy(); confs=boxes.conf.cpu().numpy()
                    cids=boxes.cls.cpu().numpy().astype(int)
                    tids=boxes.id.int().cpu().tolist() if boxes.id is not None else [None]*len(coords)
                    for box,conf,cid,tid in zip(coords,confs,cids,tids):
                        role=role_map.get(int(cid))
                        if role is None: continue
                        x1,y1,x2,y2=(int(round(v)) for v in box)
                        r={"timestamp_s":round(index/fps,3),"frame_index":index,"object_type":role,
                           "class_name":str(model.names[int(cid)]),"confidence":round(float(conf),4),
                           "track_id":int(tid) if tid is not None else None,"x1":x1,"y1":y1,"x2":x2,"y2":y2,
                           "center_x":round((x1+x2)/2,1),"center_y":float(y2)}
                        current.append(r)
                        if role=="player" and tid is not None:
                            feature=shirt_color(frame,(x1,y1,x2,y2))
                            if feature is not None: features[int(tid)].append(feature)
                rows.extend(current); by_frame[index]=current; sampled+=1
                if a.verbose and sampled%100==0: LOG.info("Analyzed %d frames, %.1f%%",sampled,100*index/max(count-1,1))
            index+=1
    finally: cap.release()
    team_ids,color_info=cluster(features); names=[a.team1_name,a.team2_name]
    summary=summarize(rows,fps,a.stride,a.possession_radius,team_ids,names,color_info)
    meta={"source":str(video),"width":w,"height":h,"fps":fps,"frame_count":count,
          "duration_seconds":count/fps if count else None,"model":a.model,"confidence_threshold":a.confidence,
          "possession_radius_px":a.possession_radius,"analyzed_frames":sampled,
          "analysis_timestamp":datetime.now().astimezone().isoformat(timespec="seconds")}
    with (out/"summary.json").open("w",encoding="utf-8") as f:
        json.dump({"schema_version":"1.0","video":meta,"summary":summary},f,indent=2,ensure_ascii=False)
    write_csv(out/"detections.csv",rows,team_ids,names)
    if not a.no_render: render(video,out/"annotated.mp4",fps,a.stride,by_frame,team_ids,names)
    LOG.info("Results: %s",out); LOG.info("Analyzed %d/%d frames; %d player IDs",sampled,count,len(features))
    return out


def main():
    a=args(); logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,format="%(levelname)s: %(message)s")
    try: run(a)
    except (FileNotFoundError,ValueError,RuntimeError) as e: LOG.error("%s",e); return 2
    except KeyboardInterrupt: LOG.warning("Interrupted; output may be incomplete."); return 130
    return 0


if __name__=="__main__": raise SystemExit(main())
