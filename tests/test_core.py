import json

import numpy as np
import pytest

from futsal import ball, config, kits, metrics, tracks
from futsal.config import AnalysisConfig, Kit
from futsal.court import Court, Pitch
from futsal.detect import Detection, class_kinds
from futsal.kits import KitClassifier

FPS = 30.0
BLUE, GREEN, PINK = Kit("Blue", (40, 55, 120), "player", "A"), Kit("Green", (85, 115, 78), "player", "B"), \
    Kit("Ref", (125, 38, 60), "referee")


CLF = KitClassifier([BLUE, GREEN, PINK], max_distance=28, min_share=0.12)


def person(frame, x, y, tid, w=20, h=60, kit=None):
    d = Detection(frame, "person", 0.9, x - w / 2, y - h, x + w / 2, y, tid)
    if kit is not None:
        d.kit_shares = np.eye(3)[kit] * 0.6
    return d


def ball_at(frame, x, y, conf=0.5):
    return Detection(frame, "ball", conf, x - 5, y - 5, x + 5, y + 5)


def frame_with_shirt(rgb, size=(80, 40)):
    img = np.full((size[0], size[1], 3), (100, 160, 200), np.uint8)  # wooden floor (BGR)
    img[10:45, 5:35] = rgb[::-1]
    return img


def test_config_loads_example():
    cfg = config.load("configs/manzano_clip.yaml")
    assert cfg.teams == ["Manzano", "Jesolo"]
    assert cfg.kit("Referee").role == "referee"
    assert len(cfg.court) >= 3


def test_config_rejects_unknown_team(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("video: x.mp4\nteams: [A, B]\nkits: [{name: K, team: C, rgb: [0, 0, 0]}]\n")
    with pytest.raises(ValueError, match="not in teams"):
        config.load(p)


def test_class_kinds_coco_names():
    assert class_kinds({0: "person", 32: "sports ball", 2: "car"}) == {0: "person", 32: "ball"}


def test_court_contains():
    court = Court([(0, 0), (100, 0), (100, 100), (0, 100)])
    assert court.contains((50, 50)) and not court.contains((150, 50))
    assert Court(None).contains((1e6, 1e6))


def test_pitch_maps_pixels_to_metres_and_running_skips_jumps():
    # a 400 x 200 px picture of a 40 x 20 m court seen straight from above: 10 px per metre
    pitch = Pitch(40, 20, [(0, 0, 0, 0), (400, 0, 40, 0), (400, 200, 40, 20), (0, 200, 0, 20)])
    assert pitch.to_m([(200, 100)])[0] == pytest.approx([20, 10], abs=1e-3)
    assert pitch.along(100, 50) == pytest.approx(.25) and pitch.along(-50, 0) == 0
    # 2 m every 2 frames at 30 fps (30 m/s) is a jump between people, 0.1 m is running
    dets = [person(0, 0, 100, 1), person(2, 1, 100, 1), person(4, 21, 100, 1), person(6, 22, 100, 1)]
    assert metrics.running_m(dets, pitch, 2, FPS) == pytest.approx(0.2, abs=1e-3)


def _jersey_row(tid, frame, team="A", h=80):
    return {"track_id": str(tid), "frame": str(frame), "team": team, "role": "player", "source": "detected",
            "x1": "100", "y1": "300", "x2": "130", "y2": str(300 + h)}


def test_jersey_vote_needs_confident_digits_and_one_shirt_at_a_time():
    from futsal import jersey
    from futsal.config import JerseyConfig
    rows = [_jersey_row(1, f) for f in range(0, 40, 4)] + [_jersey_row(2, f) for f in range(20, 60, 4)] \
        + [_jersey_row(3, f) for f in range(100, 140, 4)]
    reads = {}
    for i, r in enumerate(rows):
        k = jersey._key(r["frame"], r["x1"], r["y1"], r["x2"], r["y2"])
        tid = r["track_id"]
        # track 1 reads "10" well; track 2 overlaps it in time and reads "10" weakly; track 3 reads letters
        reads[k] = ("10", .95) if tid == "1" and i % 2 == 0 else ("10", .7) if tid == "2" and i % 5 == 0 else ("X", .9)
    out = jersey.vote(rows, reads, JerseyConfig())
    assert out["1"]["number"] == 10 and "2" not in out and "3" not in out
    assert jersey.candidates(rows + [_jersey_row(4, 0, h=40)], JerseyConfig(max_reads_per_track=3))[-1]["track_id"] == "3"
    assert jersey.is_number("7") and not jersey.is_number("123") and not jersey.is_number("1O")
    assert jersey.vote(rows, reads, JerseyConfig(roster={"A": [7, 11]})) == {}   # 10 isn't on the team sheet


def test_full_match_config_pitch_fits():
    cfg = config.load("configs/manzano_full.yaml")
    pitch = Pitch(cfg.pitch.length_m, cfg.pitch.width_m, cfg.pitch.points)
    assert pitch.residual_px < 20
    assert pitch.to_m([(963, 470)])[0][0] == pytest.approx(20, abs=.5)   # the halfway line


@pytest.mark.parametrize("kit", [BLUE, GREEN, PINK])
def test_kit_classifier_picks_shirt_colour(kit):
    clf = KitClassifier([BLUE, GREEN, PINK], max_distance=28, min_share=0.12)
    shares = clf.shares(frame_with_shirt(kit.rgb), (0, 0, 40, 80))
    assert clf.decide(shares, 1) is kit


def test_kit_classifier_rejects_floor_only():
    clf = KitClassifier([BLUE, GREEN], max_distance=28, min_share=0.12)
    floor = np.full((80, 40, 3), (100, 160, 200), np.uint8)
    assert clf.decide(clf.shares(floor, (0, 0, 40, 80)), 1) is None


def test_floor_pixels_do_not_vote():
    # dark wood is close enough to the green kit to steal votes unless it is masked out
    dark_wood = (60, 100, 130)  # BGR
    img = np.full((80, 40, 3), dark_wood, np.uint8)
    img[10:45, 15:25] = BLUE.rgb[::-1]
    floor = kits.estimate_background([np.full((50, 50, 3), dark_wood, np.uint8)], None, [BLUE, GREEN], 28, k=1)
    clf = KitClassifier([BLUE, GREEN], 28, 0.12, floor)
    shares = clf.shares(img, (0, 0, 40, 80))
    assert shares[0] == pytest.approx(1.0) and shares[1] == 0


def test_stitch_joins_fragments_of_same_player():
    a = tracks.Track(1, [person(f, 100 + f, 300, 1, kit=0) for f in range(0, 20, 2)], kit=BLUE)
    b = tracks.Track(2, [person(f, 100 + f, 300, 2, kit=0) for f in range(30, 50, 2)], kit=BLUE)
    c = tracks.Track(3, [person(f, 110 + f, 300, 3, kit=1) for f in range(30, 50, 2)], kit=GREEN)  # other team
    merged = tracks.stitch({1: a, 2: b, 3: c}, CLF, AnalysisConfig(), FPS)
    assert sorted(merged) == [1, 3]
    assert merged[1].merged_ids == [1, 2] and merged[1].kit is BLUE


def test_split_id_switch_when_shirt_changes():
    # ByteTrack ID 1 follows a blue player, then jumps to the referee after they cross
    dets = [person(f, 100, 300, 1, kit=0) for f in range(0, 20, 2)]
    dets += [person(f, 100, 300, 1, kit=2) for f in range(20, 40, 2)]
    dets[3].kit_shares = np.array([0, 0, 1.0])   # a single odd frame must not split
    parts = tracks.split_id_switches({1: tracks.Track(1, dets)}, CLF, min_run=4)
    assert [p.kit.name for p in parts.values()] == ["Blue", "Ref"]
    assert len(parts[1].detections) == 10


def test_role_limits_keep_one_goalkeeper_per_kit():
    gk = Kit("GK A", (170, 170, 180), "goalkeeper", "A")
    keeper = tracks.Track(1, [person(f, 100, 300, 1) for f in range(0, 100, 2)], kit=gk)
    staff = tracks.Track(2, [person(f, 700, 200, 2) for f in range(40, 60, 2)], kit=gk)
    later = tracks.Track(3, [person(f, 100, 300, 3) for f in range(120, 140, 2)], kit=gk)  # no overlap
    tracks.enforce_role_limits({1: keeper, 2: staff, 3: later}, overlap_tolerance=4)
    assert keeper.kit is gk and staff.kit is None and later.kit is gk


def test_fill_gaps_interpolates_short_occlusions_only():
    dets = [person(0, 100, 300, 1), person(8, 140, 300, 1), person(40, 140, 300, 1)]
    t = tracks.Track(1, dets)
    tracks.fill_gaps({1: t}, stride=2, max_gap=10)
    assert [d.frame for d in t.detections] == [0, 2, 4, 6, 8, 40]   # 32-frame gap stays empty
    assert t.detections[2].interpolated and t.detections[2].foot == pytest.approx((120, 300))
    assert not t.detections[4].interpolated


def test_ball_trajectory_rejects_jump_and_interpolates():
    frames = list(range(0, 12, 2))
    dets = [ball_at(0, 100, 100), ball_at(2, 110, 100), ball_at(2, 900, 100, conf=0.2),  # far false positive
            ball_at(8, 140, 100)]
    path = ball.trajectory(dets, frames, Court(None), AnalysisConfig(), FPS)
    assert path[2].x == 110
    assert path[4].source == "interpolated" and path[4].x == pytest.approx(120)
    assert 10 not in path


def test_ball_trajectory_ignores_a_lamp_seen_all_match():
    frames = list(range(0, 4000, 2))
    lamp = [ball_at(f, 1440, 370, conf=0.6) for f in frames]
    real = [ball_at(f, 300 + f, 400) for f in frames[:20]]
    cfg = AnalysisConfig(ball_hotspot_max_share=0.02)
    path = ball.trajectory(lamp + real, frames, Court(None), cfg, FPS)
    assert all(p.x != 1440 for p in path.values())
    assert path[10].x == 310 and len(path) == 20
    assert any(p.x == 1440 for p in ball.trajectory(lamp, frames, Court(None), AnalysisConfig(), FPS).values())


def test_possession_needs_sustained_control():
    cfg = AnalysisConfig(possession_min_s=0.1, possession_radius_px=20)
    frames = list(range(0, 20, 2))
    p_a = tracks.Track(1, [person(f, 100, 300, 1) for f in frames], kit=BLUE)
    p_b = tracks.Track(2, [person(f, 400, 300, 2) for f in frames], kit=GREEN)
    path = {f: ball.BallPoint(f, 100 if f < 10 else 400, 295, 0.5, "detected") for f in frames}
    path[10] = ball.BallPoint(10, 400, 295, 0.5, "detected")
    states = metrics.possession(frames, path, {1: p_a, 2: p_b}, cfg, FPS)
    owners = [s.possession for s in states]
    assert owners[0] is None          # control must last possession_min_s first
    assert owners[3] == "A" and owners[-1] == "B"
    assert owners[5] == "A"           # B touched at frame 10 but has not held it long enough yet


def test_count_accuracy_scores_only_in_play_frames():
    frames = list(range(0, 120, 2))
    # one Blue player on court for the first two seconds only; then a break with the court empty
    t = tracks.Track(1, [person(f, 100, 300, 1) for f in frames if f < 60], kit=BLUE)
    cfg = config.Config(video="x.mp4", teams=["A", "B"], kits=[BLUE, GREEN, PINK], expected_on_court={"Blue": 1})
    states = metrics.possession(frames, {}, {1: t}, AnalysisConfig(), FPS)
    assert metrics.summarize(cfg, FPS, 2, frames, {1: t}, 1, states)["quality"]["expected_count_match_percent"]["all"] == 50
    cfg.in_play = [(0.0, 2.0)]
    q = metrics.summarize(cfg, FPS, 2, frames, {1: t}, 1, states)["quality"]
    assert q["expected_count_match_percent"]["all"] == 100 and q["in_play_frames"] == 30
