#include "src/models/indextts/text.hpp"

#include <unicode/locid.h>
#include <unicode/normalizer2.h>
#include <unicode/regex.h>
#include <unicode/uchar.h>
#include <unicode/unistr.h>

#include <algorithm>
#include <array>
#include <fstream>
#include <functional>
#include <limits>
#include <stdexcept>
#include <utility>

namespace gufo::models::indextts {
namespace {

using icu::UnicodeString;

UnicodeString U(const std::string& utf8) { return UnicodeString::fromUTF8(utf8); }

std::string Utf8(const UnicodeString& s) {
  std::string out;
  s.toUTF8String(out);
  return out;
}

using Callback = std::function<UnicodeString(icu::RegexMatcher&)>;

UnicodeString Group(icu::RegexMatcher& m, int32_t i) {
  UErrorCode st = U_ZERO_ERROR;
  UnicodeString g = m.group(i, st);
  return U_FAILURE(st) ? UnicodeString() : g;
}

// re.sub(pattern, callback, text, flags=...)
UnicodeString Sub(const UnicodeString& text, const UnicodeString& pattern, uint32_t flags,
                  const Callback& cb) {
  UErrorCode st = U_ZERO_ERROR;
  std::unique_ptr<icu::RegexPattern> p(icu::RegexPattern::compile(pattern, flags, st));
  if (U_FAILURE(st)) {
    throw std::runtime_error("bad regex: " + Utf8(pattern));
  }
  std::unique_ptr<icu::RegexMatcher> m(p->matcher(text, st));
  UnicodeString out;
  int32_t last = 0;
  while (m->find(st) && U_SUCCESS(st)) {
    const int32_t s = m->start(st);
    const int32_t e = m->end(st);
    out.append(text, last, s - last);
    out.append(cb(*m));
    last = e;
    if (e == s) {  // empty match: copy one unit and move on (not used by our patterns)
      if (e < text.length()) {
        out.append(text.charAt(e));
      }
      last = e + 1;
      if (last > text.length()) {
        break;
      }
      m->reset(text);
      if (!m->find(last, st)) {
        break;
      }
      m->reset(text);
      m->region(last, text.length(), st);
    }
  }
  if (last < text.length()) {
    out.append(text, last, text.length() - last);
  }
  return out;
}

UnicodeString SubLiteral(const UnicodeString& text, const UnicodeString& pattern,
                         const UnicodeString& repl, uint32_t flags = 0) {
  return Sub(text, pattern, flags, [&](icu::RegexMatcher&) { return repl; });
}

// Python re.escape for our literal keys.
UnicodeString Escape(const UnicodeString& s) {
  UnicodeString out;
  for (int32_t i = 0; i < s.length(); ++i) {
    const char16_t c = s.charAt(i);
    if (c < 128 && !std::isalnum(static_cast<unsigned char>(c)) && c != '_') {
      out.append(u'\\');
    }
    out.append(c);
  }
  return out;
}

UnicodeString ReplaceAll(UnicodeString s, const UnicodeString& from, const UnicodeString& to) {
  s.findAndReplace(from, to);
  return s;
}

// --------------------------------------------------------------- numbers ---

constexpr std::array<const char*, 20> kSmall = {
    "null", "eins", "zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht", "neun",
    "zehn", "elf", "zwölf", "dreizehn", "vierzehn", "fünfzehn", "sechzehn", "siebzehn",
    "achtzehn", "neunzehn"};
constexpr std::array<const char*, 10> kTens = {"", "", "zwanzig", "dreißig", "vierzig",
                                               "fünfzig", "sechzig", "siebzig", "achtzig",
                                               "neunzig"};
constexpr std::array<const char*, 12> kMonths = {"Januar", "Februar", "März", "April",
                                                 "Mai", "Juni", "Juli", "August",
                                                 "September", "Oktober", "November", "Dezember"};

std::string UnderHundred(std::uint64_t n) {
  if (n < 20) {
    return kSmall[n];
  }
  const std::uint64_t tens = n / 10;
  const std::uint64_t ones = n % 10;
  const std::string tens_word = kTens[tens];
  if (ones == 0) {
    return tens_word;
  }
  return std::string(ones == 1 ? "ein" : kSmall[ones]) + "und" + tens_word;
}

std::string UnderThousand(std::uint64_t n) {
  if (n < 100) {
    return UnderHundred(n);
  }
  const std::uint64_t hundreds = n / 100;
  const std::uint64_t rest = n % 100;
  const std::string prefix =
      hundreds == 1 ? std::string("einhundert") : std::string(kSmall[hundreds]) + "hundert";
  return prefix + (rest != 0 ? UnderHundred(rest) : std::string());
}

std::string OrdinalToWords(std::uint64_t n) {
  switch (n) {
    case 1:
      return "erste";
    case 3:
      return "dritte";
    case 7:
      return "siebte";
    case 8:
      return "achte";
    default:
      return GermanNumberToWords(n) + (n < 20 ? "te" : "ste");
  }
}

// int(s) for a string of Unicode decimal digits; false on overflow/garbage.
bool ParseInt(const UnicodeString& s, std::uint64_t* out) {
  std::uint64_t v = 0;
  if (s.length() == 0) {
    return false;
  }
  for (int32_t i = 0; i < s.length();) {
    const UChar32 c = s.char32At(i);
    i += U16_LENGTH(c);
    const int32_t d = u_charDigitValue(c);
    if (d < 0) {
      return false;
    }
    if (v > (std::numeric_limits<std::uint64_t>::max() / 2 - 9) / 10) {
      return false;
    }
    v = v * 10 + static_cast<std::uint64_t>(d);
  }
  *out = v;
  return true;
}

std::string DigitString(const UnicodeString& digits) {
  std::string out;
  for (int32_t i = 0; i < digits.length();) {
    const UChar32 c = digits.char32At(i);
    i += U16_LENGTH(c);
    const int32_t d = u_charDigitValue(c);
    if (d < 0) {
      continue;
    }
    if (!out.empty()) {
      out += ' ';
    }
    out += kSmall[static_cast<std::size_t>(d)];
  }
  return out;
}

// Integer to words; digit-wise reading when the value does not fit.
std::string IntWords(const UnicodeString& digits) {
  std::uint64_t v = 0;
  if (ParseInt(digits, &v)) {
    return GermanNumberToWords(v);
  }
  return DigitString(digits);
}

std::string NumberToken(UnicodeString raw) {
  raw = ReplaceAll(raw, u".", u"");
  const int32_t comma = raw.indexOf(u',');
  if (comma >= 0) {
    UnicodeString whole(raw, 0, comma);
    UnicodeString fraction(raw, comma + 1);
    return IntWords(whole) + " Komma " + DigitString(fraction);
  }
  return IntWords(raw);
}

std::string YearToWords(const UnicodeString& raw) {
  std::uint64_t year = 0;
  if (!ParseInt(raw, &year)) {
    return DigitString(raw);
  }
  if (year >= 1100 && year <= 1999) {
    const std::uint64_t century = year / 100;
    const std::uint64_t rest = year % 100;
    return GermanNumberToWords(century) + "hundert" +
           (rest != 0 ? GermanNumberToWords(rest) : std::string());
  }
  return GermanNumberToWords(year);
}

// ---------------------------------------------------------- char cleanup ---

// TextNormalizer.char_rep_map / german_text._CHAR_REPLACEMENTS (dict order:
// the first key that matches at a position wins, as in the Python regex).
const std::vector<std::pair<UnicodeString, UnicodeString>>& CharReplacements() {
  static const std::vector<std::pair<UnicodeString, UnicodeString>> table = [] {
    const std::vector<std::pair<std::string, std::string>> raw = {
        {"：", ","}, {"；", ","}, {";", ","}, {"，", ","}, {"。", "."}, {"！", "!"},
        {"？", "?"}, {"\n", " "}, {"·", "-"}, {"、", ","}, {"...", "…"}, {",,,", "…"},
        {"，，，", "…"}, {"……", "…"}, {"“", "'"}, {"”", "'"}, {"\"", "'"}, {"‘", "'"},
        {"’", "'"}, {"（", "'"}, {"）", "'"}, {"(", "'"}, {")", "'"}, {"《", "'"},
        {"》", "'"}, {"【", "'"}, {"】", "'"}, {"[", "'"}, {"]", "'"}, {"—", "-"},
        {"～", "-"}, {"~", "-"}, {"「", "'"}, {"」", "'"}, {":", ","}};
    std::vector<std::pair<UnicodeString, UnicodeString>> out;
    for (const auto& [k, v] : raw) {
      out.emplace_back(U(k), U(v));
    }
    return out;
  }();
  return table;
}

UnicodeString CleanChars(const UnicodeString& text) {
  const auto& table = CharReplacements();
  UnicodeString out;
  int32_t i = 0;
  while (i < text.length()) {
    bool hit = false;
    for (const auto& [key, value] : table) {
      if (text.compare(i, key.length(), key) == 0) {
        out.append(value);
        i += key.length();
        hit = true;
        break;
      }
    }
    if (!hit) {
      out.append(text.charAt(i));
      ++i;
    }
  }
  return out;
}

bool AllInRange(const UnicodeString& s, UChar32 lo, UChar32 hi) {
  if (s.length() == 0) {
    return false;
  }
  for (int32_t i = 0; i < s.length();) {
    const UChar32 c = s.char32At(i);
    i += U16_LENGTH(c);
    if (c < lo || c > hi) {
      return false;
    }
  }
  return true;
}

bool AnyInRange(const UnicodeString& s, UChar32 lo, UChar32 hi) {
  for (int32_t i = 0; i < s.length();) {
    const UChar32 c = s.char32At(i);
    i += U16_LENGTH(c);
    if (c >= lo && c <= hi) {
      return true;
    }
  }
  return false;
}

UnicodeString ApplyPronunciationAnnotations(const UnicodeString& text) {
  return Sub(text, u"<([^|>\\n]+)\\|([^>\\n]+)>", 0, [](icu::RegexMatcher& m) {
    const UnicodeString word = Group(m, 1);
    UnicodeString pron = Group(m, 2);
    pron.toUpper(icu::Locale::getRoot());
    if (AllInRange(pron, 0x3040, 0x309F) || AllInRange(pron, 0x30A0, 0x30FF)) {
      return UnicodeString(u" ") + pron + u" ";
    }
    const UnicodeString token =
        AnyInRange(word, 0x4E00, 0x9FFF) ? u"SPECIAL_TOKEN_2" : u"SPECIAL_TOKEN_1";
    return UnicodeString(u"<|") + token + u"|>" + pron + u"<|" + token + u"|>";
  });
}

UnicodeString UpperSpecialTokens(const UnicodeString& text) {
  return Sub(text, u"<\\|([^|]+)\\|>", 0, [](icu::RegexMatcher& m) {
    UnicodeString inner = Group(m, 1);
    inner.toUpper(icu::Locale::getRoot());
    return UnicodeString(u"<|") + inner + u"|>";
  });
}

bool IsPySpace(UChar32 c) { return u_hasBinaryProperty(c, UCHAR_WHITE_SPACE) != 0 || c == 0x1C || c == 0x1D || c == 0x1E || c == 0x1F; }

UnicodeString PyStrip(const UnicodeString& s) {
  int32_t b = 0;
  int32_t e = s.length();
  while (b < e && IsPySpace(s.charAt(b))) {
    ++b;
  }
  while (e > b && IsPySpace(s.charAt(e - 1))) {
    --e;
  }
  return UnicodeString(s, b, e - b);
}

// german_text.normalize_german_v2
UnicodeString NormalizeGermanV2(UnicodeString text) {
  constexpr uint32_t kI = UREGEX_CASE_INSENSITIVE;
  UErrorCode st = U_ZERO_ERROR;
  const icu::Normalizer2* nfc = icu::Normalizer2::getNFCInstance(st);
  if (U_SUCCESS(st)) {
    text = nfc->normalize(text, st);
  }
  const std::vector<std::pair<std::string, std::string>> fillers = {
      {"呃", "äh"}, {"嗯", "hm"}, {"噢", "oh"}, {"哦", "oh"}, {"诶", "äh"}, {"欸", "äh"}, {"啊", "ah"}};
  for (const auto& [src, repl] : fillers) {
    text = ReplaceAll(text, U(src), U(" " + repl + " "));
  }
  text = SubLiteral(text, u"[\\u3400-\\u9fff]+", u" ");

  // abbreviations, longest first (stable order of the dict)
  const std::vector<std::pair<std::string, std::string>> abbreviations = {
      {"z. B.", "zum Beispiel"}, {"u. a.", "unter anderem"}, {"d. h.", "das heißt"},
      {"inkl.", "einschließlich"}, {"Prof.", "Professor"}, {"z.B.", "zum Beispiel"},
      {"bzw.", "beziehungsweise"}, {"usw.", "und so weiter"}, {"u.a.", "unter anderem"},
      {"d.h.", "das heißt"}, {"ggf.", "gegebenenfalls"}, {"ca.", "circa"},
      {"Nr.", "Nummer"}, {"Dr.", "Doktor"}};
  for (const auto& [abbr, full] : abbreviations) {
    text = SubLiteral(text, UnicodeString(u"(?<!\\w)") + Escape(U(abbr)), U(full), kI);
  }

  text = Sub(text, u"\\b([0-3]?\\d)\\.([01]?\\d)\\.(\\d{4})\\b", 0, [](icu::RegexMatcher& m) {
    std::uint64_t day = 0;
    std::uint64_t month = 0;
    if (!ParseInt(Group(m, 1), &day) || !ParseInt(Group(m, 2), &month) || day < 1 || day > 31 ||
        month < 1 || month > 12) {
      return Group(m, 0);
    }
    return U(OrdinalToWords(day) + "n " + kMonths[month - 1] + " " + YearToWords(Group(m, 3)));
  });

  UnicodeString month_names;
  for (std::size_t i = 0; i < kMonths.size(); ++i) {
    if (i != 0) {
      month_names.append(u'|');
    }
    month_names.append(Escape(U(kMonths[i])));
  }
  text = Sub(text,
             UnicodeString(u"\\b([0-3]?\\d)\\.\\s+(") + month_names + u")\\s+(\\d{4})\\b", kI,
             [](icu::RegexMatcher& m) {
               std::uint64_t day = 0;
               ParseInt(Group(m, 1), &day);
               return U(OrdinalToWords(day) + "n ") + Group(m, 2) + u" " +
                      U(YearToWords(Group(m, 3)));
             });

  const auto time_cb = [](icu::RegexMatcher& m) {
    std::uint64_t h = 0;
    std::uint64_t mi = 0;
    ParseInt(Group(m, 1), &h);
    ParseInt(Group(m, 2), &mi);
    return U(GermanNumberToWords(h) + " Uhr " + GermanNumberToWords(mi));
  };
  text = Sub(text, u"\\b([01]?\\d|2[0-3]):([0-5]\\d)(?:\\s*Uhr)?\\b", kI, time_cb);
  text = Sub(text, u"\\b([01]?\\d|2[0-3])\\.([0-5]\\d)\\s*Uhr\\b", kI, time_cb);

  const auto percent = [](icu::RegexMatcher& m) {
    return U(NumberToken(Group(m, 1)) + " Prozent");
  };
  text = Sub(text, u"%\\s*(\\d[\\d.,]*)", 0, percent);
  text = Sub(text, u"(\\d[\\d.,]*)\\s*%", 0, percent);

  const std::vector<std::pair<std::string, std::string>> currencies = {
      {"Euro", "Euro"}, {"EUR", "Euro"}, {"USD", "Dollar"}, {"GBP", "Pfund"},
      {"€", "Euro"},    {"$", "Dollar"}, {"£", "Pfund"}};
  for (const auto& [symbol, word] : currencies) {
    const UnicodeString esc = Escape(U(symbol));
    const std::string unit = word;
    text = Sub(text, UnicodeString(u"(\\d[\\d.]*)[,]([0-9]{2})\\s*") + esc + u"(?!\\w)", kI,
               [&](icu::RegexMatcher& m) {
                 const UnicodeString whole = ReplaceAll(Group(m, 1), u".", u"");
                 return U(IntWords(whole) + " " + unit + " " + IntWords(Group(m, 2)) + " Cent");
               });
    text = Sub(text, esc + u"\\s*(\\d[\\d.,]*)", kI, [&](icu::RegexMatcher& m) {
      return U(NumberToken(Group(m, 1)) + " " + unit);
    });
    text = Sub(text, UnicodeString(u"(\\d[\\d.,]*)\\s*") + esc + u"(?!\\w)", kI,
               [&](icu::RegexMatcher& m) { return U(NumberToken(Group(m, 1)) + " " + unit); });
  }

  const std::vector<std::pair<std::string, std::string>> units = {
      {"km/h", "Kilometer pro Stunde"}, {"km²", "Quadratkilometer"}, {"km2", "Quadratkilometer"},
      {"kHz", "Kilohertz"},             {"MHz", "Megahertz"},        {"GHz", "Gigahertz"},
      {"m²", "Quadratmeter"},           {"m2", "Quadratmeter"},      {"m³", "Kubikmeter"},
      {"m3", "Kubikmeter"},             {"km", "Kilometer"},         {"cm", "Zentimeter"},
      {"mm", "Millimeter"},             {"kg", "Kilogramm"},         {"mg", "Milligramm"},
      {"ml", "Milliliter"},             {"Hz", "Hertz"},             {"KB", "Kilobyte"},
      {"MB", "Megabyte"},               {"GB", "Gigabyte"},          {"TB", "Terabyte"},
      {"°C", "Grad Celsius"},           {"°F", "Grad Fahrenheit"},   {"g", "Gramm"},
      {"l", "Liter"}};
  for (const auto& [unit, word] : units) {
    const std::string expanded = word;
    text = Sub(text, UnicodeString(u"\\b(\\d[\\d.,]*)\\s*") + Escape(U(unit)) + u"(?!\\w)", kI,
               [&](icu::RegexMatcher& m) { return U(NumberToken(Group(m, 1)) + " " + expanded); });
  }

  text = Sub(text, u"\\b\\d{1,3}(?:\\.\\d{3})+\\b", 0,
             [](icu::RegexMatcher& m) { return ReplaceAll(Group(m, 0), u".", u""); });
  text = Sub(text, u"\\b(\\d+),(\\d+)\\b", 0, [](icu::RegexMatcher& m) {
    return U(IntWords(Group(m, 1)) + " Komma " + DigitString(Group(m, 2)));
  });
  text = Sub(text, u"\\b(?:1[1-9]\\d{2}|20\\d{2})\\b", 0,
             [](icu::RegexMatcher& m) { return U(YearToWords(Group(m, 0))); });
  text = Sub(text, u"\\b\\d+\\b", 0, [](icu::RegexMatcher& m) { return U(IntWords(Group(m, 0))); });
  text = Sub(text, u"\\s+([,.;:!?])", 0, [](icu::RegexMatcher& m) { return Group(m, 1); });
  text = SubLiteral(text, u"\\s+", u" ");
  return PyStrip(text);
}

// ------------------------------------------------------------- tokenizer ---

// indextts.utils.tokenizer.LANGUAGES (insertion order = LANGUAGE_DICT index)
constexpr std::array<const char*, 106> kLanguages = {
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv",
    "it", "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no",
    "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr",
    "az", "sl", "kn", "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw",
    "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu",
    "am", "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl",
    "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su", "yue", "minnan", "wuyu", "dialect",
    "zh/en", "en/zh", "common"};

bool IsLetter(UChar32 c) {
  const int8_t t = u_charType(c);
  return t == U_UPPERCASE_LETTER || t == U_LOWERCASE_LETTER || t == U_TITLECASE_LETTER ||
         t == U_MODIFIER_LETTER || t == U_OTHER_LETTER;
}
bool IsNumber(UChar32 c) {
  const int8_t t = u_charType(c);
  return t == U_DECIMAL_DIGIT_NUMBER || t == U_LETTER_NUMBER || t == U_OTHER_NUMBER;
}
bool IsSpace(UChar32 c) { return u_hasBinaryProperty(c, UCHAR_WHITE_SPACE) != 0; }

std::vector<UChar32> CodePoints(const std::string& utf8) {
  const UnicodeString s = U(utf8);
  std::vector<UChar32> out;
  for (int32_t i = 0; i < s.length();) {
    const UChar32 c = s.char32At(i);
    i += U16_LENGTH(c);
    out.push_back(c);
  }
  return out;
}

std::string FromCodePoints(const std::vector<UChar32>& cps, std::size_t b, std::size_t e) {
  UnicodeString s;
  for (std::size_t i = b; i < e; ++i) {
    s.append(cps[i]);
  }
  return Utf8(s);
}

std::string Base64Decode(const std::string& in) {
  static const std::string chars =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string out;
  int val = 0;
  int bits = -8;
  for (const char c : in) {
    if (c == '=') {
      break;
    }
    const auto pos = chars.find(c);
    if (pos == std::string::npos) {
      continue;
    }
    val = (val << 6) + static_cast<int>(pos);
    bits += 6;
    if (bits >= 0) {
      out.push_back(static_cast<char>((val >> bits) & 0xFF));
      bits -= 8;
    }
  }
  return out;
}

}  // namespace

std::string GermanNumberToWords(std::uint64_t number) {
  if (number < 1000) {
    return UnderThousand(number);
  }
  if (number < 1000000) {
    const std::uint64_t thousands = number / 1000;
    const std::uint64_t rest = number % 1000;
    const std::string prefix =
        thousands == 1 ? std::string("eintausend") : GermanNumberToWords(thousands) + "tausend";
    return prefix + (rest != 0 ? GermanNumberToWords(rest) : std::string());
  }
  struct Scale {
    std::uint64_t value;
    const char* singular;
    const char* plural;
  };
  constexpr std::array<Scale, 4> scales = {
      Scale{1000000000000000ULL, "Billiarde", "Billiarden"},
      Scale{1000000000000ULL, "Billion", "Billionen"},
      Scale{1000000000ULL, "Milliarde", "Milliarden"}, Scale{1000000ULL, "Million", "Millionen"}};
  for (const Scale& s : scales) {
    if (number >= s.value) {
      const std::uint64_t count = number / s.value;
      const std::uint64_t rest = number % s.value;
      const std::string prefix = count == 1 ? std::string("eine ") + s.singular
                                            : GermanNumberToWords(count) + " " + s.plural;
      return prefix + (rest != 0 ? " " + GermanNumberToWords(rest) : std::string());
    }
  }
  return std::to_string(number);
}

std::string PrepareGerman(const std::string& input) {
  UnicodeString text = CleanChars(U(input));
  text = NormalizeGermanV2(text);
  text.toLower(icu::Locale::getRoot());
  text = ApplyPronunciationAnnotations(text);
  text = UpperSpecialTokens(text);
  return Utf8(PyStrip(text));
}

std::int32_t LanguageToken(const std::string& lang) {
  std::string l = lang;
  std::transform(l.begin(), l.end(), l.begin(), [](unsigned char c) { return std::tolower(c); });
  for (std::size_t i = 0; i < kLanguages.size(); ++i) {
    if (l == kLanguages[i]) {
      return static_cast<std::int32_t>(i);
    }
  }
  return static_cast<std::int32_t>(kLanguages.size() - 1);  // "common"
}

std::unique_ptr<Tokenizer> Tokenizer::Load(const std::filesystem::path& path, std::string* error) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    if (error != nullptr) {
      *error = "cannot open " + path.string();
    }
    return nullptr;
  }
  auto tok = std::unique_ptr<Tokenizer>(new Tokenizer());
  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    const auto space = line.find(' ');
    if (space == std::string::npos) {
      continue;
    }
    tok->ranks_[Base64Decode(line.substr(0, space))] = std::stoi(line.substr(space + 1));
  }
  if (tok->ranks_.empty()) {
    if (error != nullptr) {
      *error = "empty tiktoken file " + path.string();
    }
    return nullptr;
  }
  // get_encoding(): specials numbered from n_vocab = len(ranks)
  std::vector<std::string> specials = {"<|endoftext|>", "<|startoftranscript|>"};
  for (std::size_t i = 0; i < 99; ++i) {
    specials.push_back(std::string("<|") + kLanguages[i] + "|>");
  }
  for (const char* e : {"ASR", "AED", "SER", "Speech", "/Speech", "BGM", "/BGM", "Laughter",
                        "/Laughter", "Applause", "/Applause", "HAPPY", "SAD", "ANGRY", "NEUTRAL"}) {
    specials.push_back(std::string("<|") + e + "|>");
  }
  for (const char* e : {"translate", "transcribe", "startoflm", "startofprev", "nospeech",
                        "notimestamps"}) {
    specials.push_back(std::string("<|") + e + "|>");
  }
  for (int i = 1; i <= 30; ++i) {
    specials.push_back("<|SPECIAL_TOKEN_" + std::to_string(i) + "|>");
  }
  for (const char* e : {"TTS/B", "TTS/O", "TTS/Q", "TTS/A", "TTS/CO", "TTS/CL", "TTS/H"}) {
    specials.push_back(std::string("<|") + e + "|>");
  }
  for (int i = 1; i <= 13; ++i) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "<|TTS/SP%02d|>", i);
    specials.emplace_back(buf);
  }
  for (int i = 0; i <= 1500; ++i) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "<|%.2f|>", i * 0.02);
    specials.emplace_back(buf);
  }
  auto next = static_cast<std::int32_t>(tok->ranks_.size());
  for (const std::string& s : specials) {
    if (!tok->special_ids_.contains(s)) {
      tok->special_ids_[s] = next;
      tok->specials_.emplace_back(s, next);
    }
    ++next;
  }
  return tok;
}

std::int32_t Tokenizer::SpecialId(const std::string& token) const {
  const auto it = special_ids_.find(token);
  return it == special_ids_.end() ? -1 : it->second;
}

void Tokenizer::BytePairEncode(const std::string& piece, std::vector<std::int32_t>* out) const {
  if (const auto it = ranks_.find(piece); it != ranks_.end()) {
    out->push_back(it->second);
    return;
  }
  // tiktoken _byte_pair_merge: repeatedly merge the adjacent pair whose
  // concatenation has the lowest rank.
  std::vector<std::size_t> starts(piece.size() + 1);
  for (std::size_t i = 0; i <= piece.size(); ++i) {
    starts[i] = i;
  }
  constexpr std::int32_t kNone = std::numeric_limits<std::int32_t>::max();
  auto rank_of = [&](std::size_t i) -> std::int32_t {  // pair (i, i+1) in `starts`
    if (i + 2 >= starts.size()) {
      return kNone;
    }
    const auto it = ranks_.find(piece.substr(starts[i], starts[i + 2] - starts[i]));
    return it == ranks_.end() ? kNone : it->second;
  };
  while (starts.size() > 2) {
    std::int32_t best = kNone;
    std::size_t best_i = 0;
    for (std::size_t i = 0; i + 2 < starts.size(); ++i) {
      const std::int32_t r = rank_of(i);
      if (r < best) {
        best = r;
        best_i = i;
      }
    }
    if (best == kNone) {
      break;
    }
    starts.erase(starts.begin() + static_cast<std::ptrdiff_t>(best_i) + 1);
  }
  for (std::size_t i = 0; i + 1 < starts.size(); ++i) {
    const auto it = ranks_.find(piece.substr(starts[i], starts[i + 1] - starts[i]));
    if (it == ranks_.end()) {
      throw std::runtime_error("tiktoken: byte sequence without rank");
    }
    out->push_back(it->second);
  }
}

void Tokenizer::EncodeOrdinary(const std::string& text, std::vector<std::int32_t>* out) const {
  // pat_str: 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
  const std::vector<UChar32> cp = CodePoints(text);
  const std::size_t n = cp.size();
  std::size_t i = 0;
  auto is_other = [](UChar32 c) { return !IsSpace(c) && !IsLetter(c) && !IsNumber(c); };
  while (i < n) {
    std::size_t j = i;
    const UChar32 c = cp[i];
    if (c == '\'' && i + 1 < n) {
      const UChar32 a = cp[i + 1];
      const UChar32 b = i + 2 < n ? cp[i + 2] : 0;
      if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) {
        j = i + 3;
      } else if (a == 's' || a == 't' || a == 'm' || a == 'd') {
        j = i + 2;
      }
    }
    if (j == i) {
      const std::size_t k = (c == ' ' && i + 1 < n) ? i + 1 : i;
      const bool lead_space = k != i;
      if (IsLetter(cp[k]) && (lead_space || IsLetter(c))) {
        j = k;
        while (j < n && IsLetter(cp[j])) {
          ++j;
        }
      } else if (IsNumber(cp[k]) && (lead_space || IsNumber(c))) {
        j = k;
        while (j < n && IsNumber(cp[j])) {
          ++j;
        }
      } else if (is_other(cp[k]) && (lead_space || is_other(c))) {
        j = k;
        while (j < n && is_other(cp[j])) {
          ++j;
        }
      } else if (IsSpace(c)) {
        std::size_t e = i;
        while (e < n && IsSpace(cp[e])) {
          ++e;
        }
        if (e == n || e - i == 1) {
          j = e == n ? n : i + 1;
        } else {
          j = e - 1;
        }
      } else {
        j = i + 1;
      }
    }
    BytePairEncode(FromCodePoints(cp, i, j), out);
    i = j;
  }
}

std::vector<std::int32_t> Tokenizer::Encode(const std::string& text) const {
  std::vector<std::int32_t> out;
  std::size_t pos = 0;
  while (pos < text.size()) {
    std::size_t best = std::string::npos;
    const std::pair<std::string, std::int32_t>* hit = nullptr;
    // earliest occurrence; at equal positions the first special in list order
    for (const auto& sp : specials_) {
      const std::size_t f = text.find(sp.first, pos);
      if (f != std::string::npos && (best == std::string::npos || f < best)) {
        best = f;
        hit = &sp;
      }
    }
    if (hit == nullptr) {
      EncodeOrdinary(text.substr(pos), &out);
      break;
    }
    if (best > pos) {
      EncodeOrdinary(text.substr(pos, best - pos), &out);
    }
    out.push_back(hit->second);
    pos = best + hit->first.size();
  }
  return out;
}

PreparedText PrepareForSynthesis(const Tokenizer& tokenizer, const std::string& input,
                                 const std::string& lang, std::int32_t max_tokens,
                                 std::int32_t capacity) {
  PreparedText result;
  result.prepared = PrepareGerman(input);
  UnicodeString text = CleanChars(U(result.prepared));
  text = ApplyPronunciationAnnotations(text);
  text = UpperSpecialTokens(text);
  result.final_text = Utf8(text);
  std::string lower_lang = lang;
  std::transform(lower_lang.begin(), lower_lang.end(), lower_lang.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  const std::string prefix = "<|" + lower_lang + "|> ";
  auto tok_len = [&](const std::string& s) {
    return static_cast<std::int32_t>(tokenizer.Encode(s).size());
  };
  const std::int32_t budget =
      std::max<std::int32_t>(1, std::min(max_tokens, capacity - 2) - tok_len(prefix));
  std::vector<std::string> segments;
  if (tok_len(result.final_text) <= budget) {
    segments.push_back(result.final_text);
  } else {
    // protected <|SPECIAL_TOKEN_n|>...<|SPECIAL_TOKEN_n|> spans stay whole
    std::vector<std::pair<UnicodeString, bool>> pieces;
    {
      UErrorCode st = U_ZERO_ERROR;
      std::unique_ptr<icu::RegexPattern> p(icu::RegexPattern::compile(
          u"<\\|SPECIAL_TOKEN_\\d+\\|>.*?<\\|SPECIAL_TOKEN_\\d+\\|>", 0, st));
      std::unique_ptr<icu::RegexMatcher> m(p->matcher(text, st));
      int32_t pos = 0;
      while (m->find(st)) {
        const int32_t s = m->start(st);
        const int32_t e = m->end(st);
        if (s > pos) {
          pieces.emplace_back(UnicodeString(text, pos, s - pos), false);
        }
        pieces.emplace_back(UnicodeString(text, s, e - s), true);
        pos = e;
      }
      if (pos < text.length()) {
        pieces.emplace_back(UnicodeString(text, pos), false);
      }
    }
    static const UnicodeString kCut = U("，。！？、；：,.!?;:\n");
    std::vector<std::string> chunks;
    for (const auto& [piece, atomic] : pieces) {
      if (atomic) {
        chunks.push_back(Utf8(piece));
        continue;
      }
      std::vector<UnicodeString> parts;
      UnicodeString cur;
      for (int32_t i = 0; i < piece.length();) {
        const UChar32 c = piece.char32At(i);
        i += U16_LENGTH(c);
        cur.append(c);
        if (kCut.indexOf(c) >= 0) {
          parts.push_back(cur);
          cur.remove();
        }
      }
      if (cur.length() > 0) {
        parts.push_back(cur);
      }
      for (const UnicodeString& part : parts) {
        const std::string p8 = Utf8(part);
        if (tok_len(p8) <= budget) {
          chunks.push_back(p8);
          continue;
        }
        UnicodeString acc;
        for (int32_t i = 0; i < part.length();) {
          const UChar32 c = part.char32At(i);
          i += U16_LENGTH(c);
          UnicodeString trial = acc;
          trial.append(c);
          if (acc.length() > 0 && tok_len(Utf8(trial)) > budget) {
            chunks.push_back(Utf8(acc));
            acc.remove();
            acc.append(c);
          } else {
            acc = trial;
          }
        }
        if (acc.length() > 0) {
          chunks.push_back(Utf8(acc));
        }
      }
    }
    std::string cur;
    for (const std::string& c : chunks) {
      if (!cur.empty() && tok_len(cur + c) > budget) {
        segments.push_back(cur);
        cur = c;
      } else {
        cur += c;
      }
    }
    if (!cur.empty()) {
      segments.push_back(cur);
    }
    if (segments.empty()) {
      segments.push_back(result.final_text);
    }
  }
  for (const std::string& seg : segments) {
    std::vector<std::int32_t> ids = tokenizer.Encode(prefix + seg);
    ids.push_back(1);
    result.ids.push_back(std::move(ids));
  }
  result.segments = std::move(segments);
  return result;
}

}  // namespace gufo::models::indextts
