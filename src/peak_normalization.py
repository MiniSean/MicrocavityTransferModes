from src.peak_identifier import PeakCollection
from src.peak_relation import LabeledPeakCollection, LabeledPeak


class NormalizedPeak(LabeledPeak):
    """LabeledPeak with additional normalized positioning"""

    def __init__(self, labeled_peak: LabeledPeak, **kwargs):
        super().__init__(labeled_peak, labeled_peak.get_longitudinal_mode_id, labeled_peak.get_transverse_mode_id)


# Adds additional normalization functionality to the labeled peak collection
class NormalizedPeakCollection(LabeledPeakCollection):
    """LabeledPeakCollection with additional normalization functionality to the labeled peak collection"""

    def __init__(self, optical_mode_collection: PeakCollection):
        super().__init__(optical_mode_collection)


