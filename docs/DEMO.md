# Demo script

A 90-second walkthrough of Gilto, plus a short pitch for interviews. Every question below was run against the real app with the sample files in `docs/demo/`, and the answers shown are what it returned.

## Before you record

- Use the two sample files: `docs/demo/acme-policies.pdf` (a 3-page text PDF) and `docs/demo/sale-notice.png` (an image with no text layer, so it needs OCR).
- Start from a clean account: open **Account** in the sidebar and use **Delete my data**, so the library is empty and the first screen shows the three setup steps.
- Window about 1280 x 720, browser zoom 100%, dark mode, no bookmarks bar. Close other tabs and any notifications.
- If you use a free Gemini key, wait a minute between takes. Rate limits show up as "rate limit or quota was reached".
- Have the Sources panel ready to open, because it is the key moment.

## Script

| Time | You do | You say | Viewers see |
|---|---|---|---|
| 0:00 | Show the sign-in card, then sign in. | "Gilto lets you chat with your own documents. It is invite-only, so I sign in with an approved Google account." | The "Restricted access" card, then the workspace with the three setup steps. |
| 0:10 | Tick the consent box. Drag `acme-policies.pdf` into the uploader and click **Add to library**. | "Before the first upload it asks for consent, because the text goes to Google's Gemini API. Now I add a PDF." | A progress bar, then a success note and the document card: 3 pages, 3 passages. |
| 0:22 | Click **What is this document about?** | "These starter questions ask for a summary. It reads the whole document, not just a few passages." | "Reading part 1 of 1", then a bulleted summary with citation chips. |
| 0:35 | Type: **How long is the warranty on accessories?** | "Now a specific question." | "Accessories carry a 6 month warranty" with a chip `acme-policies.pdf · p.2`. |
| 0:45 | Open **Sources**. | "Every answer shows the exact passage it came from, with the page and a relevance score. If the model invents a citation, Gilto removes it." | The source card with page 2, the passage text and a relevance score. |
| 0:55 | Type: **Who won the 2018 football world cup?** | "If the answer is not in my files, it says so instead of guessing." | "I couldn't find that in the uploaded documents." with no sources. |
| 1:05 | Drag in `sale-notice.png` and click **Add to library**. Then ask: **When does the spring sale end, and do members get a discount?** | "This one is a picture with no text. Gilto reads it with OCR. Tesseract runs first, and Gemini vision takes over when Tesseract is unsure or the page is Hindi." | "The spring sale ends on 14 April, and members get an extra 15 percent off", citing `sale-notice.png · p.1` with an `ocr` or `vision` tag. |
| 1:20 | Point at the **System status** panel and the chat list. | "Usage limits, saved chats and a delete-my-data button are built in. Each user's library is private." | The usage bars and the chat history in the sidebar. |
| 1:30 | Stop. | | |

If a question is slow, do not stop. Keep talking about what is happening ("it is searching with vectors and keywords, then checking the citations") and cut the pause when you edit.

## Expected answers

Use these to check a take quickly.

| Question | Answer |
|---|---|
| How many days do I have to return an item? | 30 days, `acme-policies.pdf · p.1` |
| How long is the warranty on accessories? | 6 months, `p.2` |
| When is support open? | Monday to Friday, 9:00 to 17:00 CET, `p.3` |
| How much is express shipping? | 12 euros, arrives in 2 business days, `p.3` |
| When does the spring sale end, and do members get a discount? | 14 April, extra 15 percent off for members, `sale-notice.png · p.1` |
| Who won the 2018 football world cup? | "I couldn't find that in the uploaded documents." |

## Recording

- **Windows:** press `Win + G` for the Game Bar and record the browser window, or use [ScreenToGif](https://www.screentogif.com/) to capture straight to a GIF.
- **Length and size:** a GIF over about 10 MB loads slowly on GitHub. For a 90-second clip, export an MP4 instead.
- **Put it in the README:** open any issue on the repo, drag the MP4 into the comment box, wait for the upload, and copy the generated `https://github.com/user-attachments/...` link. Paste that link on its own line in `README.md` and GitHub shows it as a video player. You do not need to submit the issue.
- **Thumbnail fallback:** if you keep the video elsewhere, link `docs/screenshots/chat.png` to it.

## Pitch

**Thirty seconds.** "Gilto is a document chatbot that only answers from your files and cites the page. It reads PDFs, scans and images, including Hindi, finds passages with both vector and keyword search, and checks every citation against what it actually retrieved. It has Google sign-in, private libraries, usage limits and 243 automated tests, and it runs on Streamlit with Gemini and Chroma."

**Two minutes: what to point at.**
- *Grounding.* A strict prompt, a similarity threshold, and a citation checker. An out-of-scope question is refused without even calling the model.
- *Retrieval.* Vector search misses exact terms such as invoice numbers, so keyword search runs beside it and the two lists are merged by rank fusion.
- *Reliability.* Gemini rate-limits and sometimes returns server errors. Calls retry, then fall back to a second model and then to a cheaper one, so a question rarely fails.
- *Safety.* Document text is treated as data. A test checks that an instruction hidden inside a PDF is ignored. A commit hook and CI scan for leaked keys, and the scanner has already stopped real credentials from being committed once.
- *Privacy.* Each user's data is isolated, there is a consent step, a delete-my-data button, and automatic removal of inactive accounts.

**Be honest about the limits.** Handwriting is not read. Hindi OCR has not been measured on real scans. On the free Streamlit host, data disappears when the app sleeps. The similarity threshold was set from a small sample. Saying this up front is a strength: it shows you know what you measured and what you did not.

**Questions you may get.**
- *Why Chroma and not a hosted vector database?* One user's data is small, a local store is simple, and each user already gets a separate collection. A hosted database is the next step if it needs to persist and scale.
- *How do you stop hallucinations?* You cannot fully, so Gilto limits the damage: answer only from retrieved text, refuse when nothing matches, verify citations, and show the source passage so the reader can check.
- *What would you do next?* Keep original files so a citation opens the PDF at that page, add reranking, and move to permanent storage.
