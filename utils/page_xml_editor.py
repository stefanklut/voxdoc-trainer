import datetime
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.joinpath("..")))
from utils.logging_utils import get_logger
from utils.tempdir import AtomicFileName


class Coords(ET.Element):
    def __init__(self, points: np.ndarray, **kwargs):
        super().__init__("Coords", **kwargs)
        self.points = points

    @property
    def points(self) -> np.ndarray:
        str_points = self.attrib["points"]
        points = np.array([i.split(",") for i in str_points]).astype(np.int32)
        return points

    @points.setter
    def points(self, value: np.ndarray):
        assert value.shape[1] == 2, f"Expected 2D array, got {value.shape}"
        str_coords = ""
        for coords in value:
            str_coords += f" {round(coords[0])},{round(coords[1])}"
        self.attrib["points"] = str_coords.strip()


class Baseline(Coords):
    def __init__(self, points: np.ndarray, **kwargs):
        super().__init__(points, **kwargs)
        self.tag = "Baseline"


class _Polygon(ET.Element):
    def __init__(self, points: np.ndarray, id: Optional[str] = None, custom: Optional[str] = None, **kwargs):
        super().__init__("_Polygon", **kwargs)
        self.append(Coords(points))
        if id is not None:
            self.attrib["id"] = id
        if custom is not None:
            self.attrib["custom"] = custom


class TextLine(_Polygon):
    def __init__(self, points: np.ndarray, reading_order: Optional[int] = None, **kwargs):
        super().__init__(points, **kwargs)
        self.tag = "TextLine"
        self.logger = get_logger()
        self.reading_order = reading_order

    @property
    def reading_order(self) -> Optional[int]:
        try:
            re_match = re.match(r".*readingOrder {index:(\d+);.*}", self.attrib["custom"])
        except KeyError:
            self.logger.warning(f"No reading order defined for {self.attrib['id']}")
            return None
        if re_match is None:
            self.logger.warning(f"No reading order defined for {self.attrib['id']}")
            return None
        reading_order_index = re_match.group(1)

        return int(reading_order_index)

    @reading_order.setter
    def reading_order(self, value: Optional[int]):
        if value is not None:
            self.attrib["custom"] = f"readingOrder {{index:{value};}}"


class TextEquiv(ET.Element):
    def __init__(self, value: str, **kwargs):
        super().__init__("TextEquiv", **kwargs)
        unicode = ET.SubElement(self, "Unicode")
        unicode.text = value


class Region(_Polygon):
    def __init__(self, points: np.ndarray, region_type: Optional[str] = None, **kwargs):
        super().__init__(points, **kwargs)
        self.logger = get_logger()

        self.tag = "Region"
        self.region_type = region_type

    @property
    def region_type(self) -> Optional[str]:
        try:
            # The capture stops at the first ';' so a custom attribute holding
            # several entries (e.g. 'structure {type:x;} readingOrder {index:1;}')
            # still yields the bare type.
            re_match = re.match(r".*structure {.*type:([^;]*);.*}", self.attrib["custom"])
        except KeyError:
            self.logger.warning(f"No region type defined for {self.attrib['id']}")
            return None
        if re_match is None:
            self.logger.warning(f"No region type defined for {self.attrib['id']}")
            return None
        region_type = re_match.group(1)

        return region_type

    @region_type.setter
    def region_type(self, value: Optional[str]):
        if value is not None:
            self.attrib["custom"] = f"structure {{type:{value};}}"

    @classmethod
    def with_tag(cls, tag: str, points: np.ndarray, region_type: Optional[str] = None, **kwargs):
        region = cls(points, region_type, **kwargs)
        region.tag = tag
        return region


class PcGts(ET.Element):
    def __init__(self, **kwargs):
        super().__init__("PcGts", **kwargs)
        self.attrib = {
            "xmlns": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15",
            "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "xsi:schemaLocation": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15 http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15/pagecontent.xsd",
        }


class Metadata(ET.Element):
    def __init__(self, **kwargs):
        super().__init__("Metadata", **kwargs)


class Creator(ET.Element):
    def __init__(self, text, **kwargs):
        super().__init__("Creator", **kwargs)
        self.text = text


class Created(ET.Element):
    def __init__(self, **kwargs):
        super().__init__("Created", **kwargs)
        self.text = datetime.datetime.today().strftime("%Y-%m-%dT%X")


class LastChange(ET.Element):
    def __init__(self, **kwargs):
        super().__init__("LastChange", **kwargs)
        self.text = datetime.datetime.today().strftime("%Y-%m-%dT%X")


class Page(ET.Element):
    def __init__(self, image_filename: str, image_width: int, image_height: int, **kwargs):
        super().__init__("Page", **kwargs)
        self.attrib = {
            "imageFilename": image_filename,
            "imageWidth": str(image_width),
            "imageHeight": str(image_height),
        }


class PageXML(ET.ElementTree):
    def __init__(self, **kwargs):
        super().__init__(element=PcGts(), **kwargs)

    def save_xml(self, filepath: Path):
        """write out XML file of current PAGE data"""
        self._indent(self.getroot())
        with AtomicFileName(filepath) as path:
            super().write(path, encoding="UTF-8", xml_declaration=True)

    def _indent(self, elem, level=0):
        """
        Function borrowed from:
            http://effbot.org/zone/element-lib.htm
        """
        i = "\n" + level * "  "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for elem in elem:
                self._indent(elem, level + 1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i


def is_low_confidence(confidence: Optional[float], min_confidence: Optional[float]) -> bool:
    """Report whether recognised text falls below a confidence threshold.

    Handwritten-text recognition reports a confidence per line, and a corpus can
    carry a large minority of lines the recogniser was unsure about. Those lines
    make poor training targets, so creators let callers drop them. Text with no
    recorded confidence is treated as trustworthy rather than discarded.

    Args:
        confidence (Optional[float]): The line's reported confidence, or None.
        min_confidence (Optional[float]): The threshold to enforce, or None to keep
            everything.

    Returns:
        bool: True when the line should be dropped.
    """
    if min_confidence is None or confidence is None:
        return False
    return confidence < min_confidence


def add_min_conf_argument(parser: Any, default: Optional[float] = None) -> None:
    """Add a ``--min-conf`` flag to an argparse parser.

    Shared by every creator that uses recognised text as a training target, so the
    flag and its help text stay identical everywhere.

    Args:
        parser (Any): The ``argparse.ArgumentParser`` to extend.
        default (Optional[float]): The default threshold. None keeps every line, so
            existing commands keep producing the same datasets.
    """
    parser.add_argument(
        "--min-conf",
        dest="min_conf",
        type=float,
        default=default,
        metavar="CONF",
        help=(
            "Drop text lines whose recognised transcription has a TextEquiv confidence below CONF "
            "(a value in [0, 1]). Handwritten-text recognition is often confidently wrong, and those "
            "lines teach the model to make the same mistakes. Lines with no recorded confidence are "
            "kept. Unset by default, which keeps every line."
        ),
    )


class PageXMLEditor(PageXML):
    """Class to process PAGE xml files"""

    def __init__(self, filepath: Optional[Path | str] = None):
        self.logger = get_logger()
        self.XMLNS = {
            "xmlns": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15",
            "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "xsi:schemaLocation": " ".join(
                [
                    "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15",
                    " http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15/pagecontent.xsd",
                ]
            ),
        }

        ET.register_namespace("", self.XMLNS["xmlns"])

        self.filepath = Path(filepath) if filepath is not None else None
        if self.filepath is not None:
            if self.filepath.exists():
                self.parse(self.filepath)
                root = self.getroot()
                if root is None:
                    raise ValueError("Root element not found in the XML tree.")
                # HACK Remove namespace from tree
                default_namespace = root.tag.split("}")[0] + "}"
                for elem in self.iter():
                    name_space = elem.tag.split("}")[0] + "}"
                    if name_space and name_space != default_namespace:
                        raise ValueError(f"Found unexpected namespace {name_space}")
                    if name_space == default_namespace:
                        elem.tag = elem.tag.split("}")[1]
            else:
                raise FileNotFoundError(f"File {filepath} does not exist")

        if filepath is None:
            super().__init__()
            self.add_metadata()

        self.size: Optional[tuple[int, int]] = None

    def set_size(self, size: tuple[int, int]):
        self.size = size

    def get_size(self):
        """
        Get Image size defined on XML file
        """
        if self.size is not None:
            return self.size

        page = self.find(".//Page")
        if page is None:
            raise ValueError("Page element is missing in the XML file.")

        img_width_str = page.get("imageWidth")
        if img_width_str is None:
            raise ValueError("imageWidth attribute is missing in the XML file.")
        img_width = int(img_width_str)
        img_height_str = page.get("imageHeight")
        if img_height_str is None:
            raise ValueError("imageHeight attribute is missing in the XML file.")
        img_height = int(img_height_str)
        self.size = (img_height, img_width)

        return self.size

    def get_region_type(self, element):
        """
        Returns the type of element
        """
        try:
            # The capture stops at the first ';' so a custom attribute holding
            # several entries (e.g. 'structure {type:x;} readingOrder {index:1;}')
            # still yields the bare type.
            re_match = re.match(r".*structure {.*type:([^;]*);.*}", element.attrib["custom"])
        except KeyError:
            self.logger.warning(f"No region type defined for {self.get_id(element)} at {self.filepath}")
            return None
        if re_match is None:
            self.logger.warning(f"No region type defined for {self.get_id(element)} at {self.filepath}")
            return None
        e_type = re_match.group(1)

        return e_type

    def get_id(self, element) -> str:
        """
        get Id of current element
        """
        return str(element.attrib.get("id"))

    def get_zones(self, region_names):
        to_return = {}
        idx = 0
        for element in region_names:
            for node in self.iterfind(f".//{element}"):
                to_return[idx] = {
                    "coords": self.get_coords(node),
                    "type": self.get_region_type(node),
                    "id": self.get_id(node),
                }
                idx += 1
        if to_return:
            return to_return
        else:
            return None

    def get_coords(self, element: ET.Element) -> np.ndarray:
        coords_element = element.find("Coords")
        if coords_element is None:
            raise ValueError(f"'Coords' element not found in the provided element: {element.tag}")
        points_attr = coords_element.attrib.get("points")
        if points_attr is None:
            raise ValueError(f"'points' attribute is missing in the 'Coords' element: {coords_element.tag}")
        str_coords = points_attr.split()
        return np.array([i.split(",") for i in str_coords]).astype(np.int32)

    def iter_class_coords(self, element, class_dict):
        for node in self.iterfind(f".//{element}"):
            element_type = self.get_region_type(node)
            if element_type is None or element_type not in class_dict:
                self.logger.warning(f'Element type "{element_type}" undefined in class dict {self.filepath}')
                continue
            element_class = class_dict[element_type]
            element_coords = self.get_coords(node)

            # Ignore lines
            if element_coords.shape[0] < 3:
                continue

            yield element_class, element_coords

    def iter_baseline_coords(self):
        for node in self.iterfind(".//Baseline"):
            str_coords = node.attrib.get("points")
            # Ignoring empty baselines
            if str_coords is None:
                continue
            split_str_coords = str_coords.split()
            # Ignoring empty baselines
            if len(split_str_coords) == 0:
                continue
            # HACK Doubling single value baselines (otherwise they are not drawn)
            if len(split_str_coords) == 1:
                split_str_coords = split_str_coords * 2  # Double list [value]*2 for cv2.polyline
            coords = np.array([i.split(",") for i in split_str_coords]).astype(np.int32)
            yield coords

    def iter_class_baseline_coords(self, element, class_dict):
        for class_node in self.iterfind(f".//{element}"):
            element_type = self.get_region_type(class_node)
            if element_type is None or element_type not in class_dict:
                self.logger.warning(f'Element type "{element_type}" undefined in class dict {self.filepath}')
                continue
            element_class = class_dict[element_type]
            for baseline_node in class_node.iterfind(".//Baseline"):
                str_coords = baseline_node.attrib.get("points")
                # Ignoring empty baselines
                if str_coords is None:
                    continue
                split_str_coords = str_coords.split()
                # Ignoring empty baselines
                if len(split_str_coords) == 0:
                    continue
                # HACK Doubling single value baselines (otherwise they are not drawn)
                if len(split_str_coords) == 1:
                    split_str_coords = split_str_coords * 2  # Double list [value]*2 for cv2.polyline
                coords = np.array([i.split(",") for i in split_str_coords]).astype(np.int32)
                yield element_class, coords

    def iter_text_line_coords(self):
        for node in self.iterfind(".//TextLine"):
            coords = self.get_coords(node)
            yield coords

    def add_metadata(self):
        metadata = Metadata()
        metadata.append(Creator("laypa"))
        metadata.append(Created())
        metadata.append(LastChange())
        root = self.getroot()
        if root is None:
            raise ValueError("Root element not found in the XML tree.")
        root.append(metadata)
        return metadata

    def add_page(
        self,
        image_filename: str,
        image_height: int,
        image_width: int,
    ):
        page = Page(image_filename, image_width, image_height)
        root = self.getroot()
        if root is None:
            raise ValueError("Root element not found in the XML tree.")
        root.append(page)
        return page

    def get_text(self, element):
        """
        get Text defined for element (prefers the Unicode child of TextEquiv)
        """
        text_node = element.find("TextEquiv")
        if text_node is None:
            self.logger.info(f"No Text node found for line {self.get_id(element)} at {self.filepath}")
            return ""
        else:
            # Prefer the Unicode child; fall back to the first child (e.g. PlainText).
            child_node = text_node.find("Unicode")
            if child_node is None:
                child_node = text_node.find("*")
            if child_node is None or child_node.text is None:
                self.logger.info(f"No text found in line {self.get_id(element)} at {self.filepath}")
                return ""
            else:
                return child_node.text

    def get_text_confidence(self, element: ET.Element) -> Optional[float]:
        """Read the confidence the recogniser reported for an element's text.

        Args:
            element (ET.Element): A ``TextLine`` or ``TextRegion`` element.

        Returns:
            Optional[float]: The ``TextEquiv`` ``conf`` attribute in ``[0, 1]``, or
                None when the element carries no text or no confidence.
        """
        text_node = element.find("TextEquiv")
        if text_node is None:
            return None
        raw = text_node.attrib.get("conf")
        if raw is None:
            return None
        try:
            return float(raw)
        except ValueError:
            self.logger.info(f"Unparseable text confidence '{raw}' on {self.get_id(element)} at {self.filepath}")
            return None

    def get_transcription(self):
        """Extracts text from each line on the XML file"""
        data = {}
        for element in self.iterfind(".//TextRegion"):
            r_id = self.get_id(element)
            for line in element.iterfind(".//TextLine"):
                l_id = self.get_id(line)
                data["_".join([r_id, l_id])] = self.get_text(line)

        return data

    def get_transcription_text(self, min_confidence: Optional[float] = None) -> str:
        """Extract the full transcription as a single unicode string.

        Regions are ordered by their reading order and their lines by theirs, so
        the text of one region always stays together. Reading order indices are
        scoped to the element that declares them: corpora that number lines from
        zero inside every region (rather than once per page) would otherwise have
        their paragraphs interleaved line by line. Lines without a reading order
        keep their document order and are placed last within their region.

        Args:
            min_confidence (Optional[float]): Drop lines whose recognised text has
                a ``TextEquiv`` confidence below this value. Lines with no
                recorded confidence are kept. None keeps every line.

        Returns:
            str: The transcription text, or an empty string if there is none.
        """
        ordered_regions = self._ordered_regions()
        lines: list[tuple[int, Optional[int], int, str]] = []
        for region_index, region in enumerate(ordered_regions):
            for line_index, line in enumerate(region.iterfind(".//TextLine")):
                text = self.get_text(line).strip()
                if not text:
                    continue
                if min_confidence is not None:
                    confidence = self.get_text_confidence(line)
                    if confidence is not None and confidence < min_confidence:
                        continue
                reading_order = self._parse_reading_order(line)
                lines.append((region_index, reading_order, line_index, text))
        lines.sort(key=lambda item: (item[0], item[1] is None, item[1] if item[1] is not None else 0, item[2]))
        return "\n".join(text for _, _, _, text in lines)

    def _ordered_regions(self) -> list[ET.Element]:
        """Return every text region, ordered by its page-level reading order.

        Regions that carry no reading order keep their document order and are
        placed after the ordered ones.

        Returns:
            list[ET.Element]: The regions to read lines from.
        """
        regions: list[tuple[Optional[int], int, ET.Element]] = []
        for index, region in enumerate(self.iterfind(".//TextRegion")):
            regions.append((self._parse_reading_order(region), index, region))
        regions.sort(key=lambda item: (item[0] is None, item[0] if item[0] is not None else 0, item[1]))
        return [region for _, _, region in regions]

    def _parse_reading_order(self, element: ET.Element) -> Optional[int]:
        """Parse the reading order index from an element's ``custom`` attribute.

        The index is only meaningful among siblings that share the same parent,
        so it must never be compared across regions.

        Args:
            element (ET.Element): The element to parse.

        Returns:
            Optional[int]: The reading order index, or None if not defined.
        """
        custom = element.attrib.get("custom")
        if custom is None:
            return None
        re_match = re.match(r".*readingOrder {index:(\d+);.*}", custom)
        if re_match is None:
            return None
        return int(re_match.group(1))
