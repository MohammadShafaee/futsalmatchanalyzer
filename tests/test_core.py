import numpy as np
import pytest

from futsal import ball, config, kits, metrics, tracks
from futsal.config import AnalysisConfig, Kit
from futsal.court import Court
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


def test_ball_trajectory_rejects_jump_and_interpolates():
    frames = list(range(0, 12, 2))
    dets = [ball_at(0, 100, 100), ball_at(2, 110, 100), ball_at(2, 900, 100, conf=0.2),  # far false positive
            ball_at(8, 140, 100)]
    path = ball.trajectory(dets, frames, Court(None), AnalysisConfig(), FPS)
    assert path[2].x == 110
    assert path[4].source == "interpolated" and path[4].x == pytest.approx(120)
    assert 10 not in path


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
