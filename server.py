import os
import shutil
import pickle
import json
from functools import lru_cache
from typing import Optional
from fastapi import FastAPI, Request, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from deep_translator import GoogleTranslator

from dotenv import load_dotenv

from reader3 import Book, BookMetadata, ChapterContent, TOCEntry, process_epub, save_to_pickle

# Load environment variables
load_dotenv()

# --- CONFIG ---
BOOKS_DIR = "."


app = FastAPI()
templates = Jinja2Templates(directory="templates")



@lru_cache(maxsize=10)
def load_book_cached(folder_name: str) -> Optional[Book]:
    """
    Loads the book from the pickle file.
    Cached so we don't re-read the disk on every click.
    """
    file_path = os.path.join(BOOKS_DIR, folder_name, "book.pkl")
    if not os.path.exists(file_path):
        return None

    try:
        with open(file_path, "rb") as f:
            book = pickle.load(f)
        return book
    except Exception as e:
        print(f"Error loading book {folder_name}: {e}")
        return None

def get_book_status(book_id: str) -> dict:
    """Get book status (finished, etc.) from status.json"""
    status_file = os.path.join(BOOKS_DIR, book_id, "status.json")
    if os.path.exists(status_file):
        try:
            with open(status_file, "r") as f:
                return json.load(f)
        except:
            pass
    return {"finished": False}

def set_book_status(book_id: str, status: dict):
    """Save book status to status.json"""
    status_file = os.path.join(BOOKS_DIR, book_id, "status.json")
    try:
        with open(status_file, "w") as f:
            json.dump(status, f)
    except Exception as e:
        print(f"Error saving status: {e}")

@app.get("/", response_class=HTMLResponse)
async def library_view(request: Request):
    """Lists all available processed books."""
    books = []

    # Scan directory for folders ending in '_data' that have a book.pkl
    if os.path.exists(BOOKS_DIR):
        for item in os.listdir(BOOKS_DIR):
            item_path = os.path.join(BOOKS_DIR, item)
            if os.path.isdir(item_path) and item.endswith("_data"):
                # Try to load it to get the title
                book = load_book_cached(item)
                if book:
                    status = get_book_status(item)
                    books.append({
                        "id": item,
                        "title": book.metadata.title,
                        "author": ", ".join(book.metadata.authors),
                        "chapters": len(book.spine),
                        "finished": status.get("finished", False)
                    })

    return templates.TemplateResponse("library.html", {"request": request, "books": books})

@app.post("/upload")
async def upload_book(file: UploadFile = File(...)):
    """Uploads and processes an EPUB file."""
    if not file.filename or not file.filename.lower().endswith(".epub"):
        raise HTTPException(status_code=400, detail="Only .epub files are allowed")
    
    # Sanitize filename to prevent directory traversal or weird chars
    safe_filename = "".join([c for c in file.filename if c.isalpha() or c.isdigit() or c in '._-']).strip()
    if not safe_filename:
        safe_filename = "uploaded_book.epub"
    temp_filename = f"temp_{safe_filename}"
    
    try:
        # Save uploaded file temporarily
        with open(temp_filename, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
            
        # Determine output directory (same logic as reader3.py CLI)
        # We want the folder name to be based on the filename, e.g. "mybook.epub" -> "mybook_data"
        base_name = os.path.splitext(safe_filename)[0]
        out_dir = os.path.join(BOOKS_DIR, f"{base_name}_data")
        
        # Process the EPUB
        # process_epub and save_to_pickle are imported from reader3.py
        print(f"Processing uploaded file: {temp_filename} -> {out_dir}")
        book = process_epub(temp_filename, out_dir)
        save_to_pickle(book, out_dir)
        
        return {"success": True, "message": f"Book '{book.metadata.title}' uploaded successfully!"}
        
    except Exception as e:
        print(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process book: {str(e)}")
    finally:
        # Clean up temp file
        if os.path.exists(temp_filename):
            try:
                os.remove(temp_filename)
            except:
                pass

@app.get("/read/{book_id}", response_class=HTMLResponse)
async def redirect_to_first_chapter(request: Request, book_id: str):
    """Helper to just go to chapter 0."""
    return await read_chapter(request=request, book_id=book_id, chapter_index=0)

@app.get("/read/{book_id}/{chapter_index}", response_class=HTMLResponse)
async def read_chapter(request: Request, book_id: str, chapter_index: int):
    """The main reader interface."""
    book = load_book_cached(book_id)
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")

    if chapter_index < 0 or chapter_index >= len(book.spine):
        raise HTTPException(status_code=404, detail="Chapter not found")

    current_chapter = book.spine[chapter_index]

    # Calculate Prev/Next links
    prev_idx = chapter_index - 1 if chapter_index > 0 else None
    next_idx = chapter_index + 1 if chapter_index < len(book.spine) - 1 else None

    return templates.TemplateResponse("reader.html", {
        "request": request,
        "book": book,
        "current_chapter": current_chapter,
        "chapter_index": chapter_index,
        "book_id": book_id,
        "prev_idx": prev_idx,
        "next_idx": next_idx
    })

@app.get("/read/{book_id}/images/{image_name}")
async def serve_image(book_id: str, image_name: str):
    """
    Serves images specifically for a book.
    The HTML contains <img src="images/pic.jpg">.
    The browser resolves this to /read/{book_id}/images/pic.jpg.
    """
    # Security check: ensure book_id is clean
    safe_book_id = os.path.basename(book_id)
    safe_image_name = os.path.basename(image_name)

    img_path = os.path.join(BOOKS_DIR, safe_book_id, "images", safe_image_name)

    if not os.path.exists(img_path):
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(img_path)

class TranslationRequest(BaseModel):
    text: str
    target_lang: str = "id"  # Default to Indonesian as requested context implies, or generic 'en'

@app.post("/translate")
async def translate_text(request: TranslationRequest):
    try:
        # Use Google Translator
        translator = GoogleTranslator(source='auto', target=request.target_lang)
        translated = translator.translate(request.text)
        return {"translated_text": translated}
    except Exception as e:
        print(f"Translation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))



@app.delete("/books/{book_id}")
async def delete_book(book_id: str):
    """Deletes a book and its data directory."""
    # Security check: ensure book_id is clean and exists in BOOKS_DIR
    safe_book_id = os.path.basename(book_id)
    book_path = os.path.join(BOOKS_DIR, safe_book_id)
    
    if not os.path.exists(book_path) or not os.path.isdir(book_path):
        raise HTTPException(status_code=404, detail="Book not found")
        
    try:
        shutil.rmtree(book_path)
        # Clear cache to prevent serving deleted book
        load_book_cached.cache_clear() 
        return {"success": True, "message": f"Book '{safe_book_id}' deleted successfully"}
    except Exception as e:
        print(f"Delete error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete book: {str(e)}")

class StatusRequest(BaseModel):
    finished: bool

@app.post("/books/{book_id}/status")
async def update_book_status(book_id: str, request: StatusRequest):
    """Update book status (finished/unfinished)"""
    safe_book_id = os.path.basename(book_id)
    book_path = os.path.join(BOOKS_DIR, safe_book_id)
    
    if not os.path.exists(book_path) or not os.path.isdir(book_path):
        raise HTTPException(status_code=404, detail="Book not found")
    
    try:
        set_book_status(safe_book_id, {"finished": request.finished})
        return {"success": True, "finished": request.finished}
    except Exception as e:
        print(f"Status update error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("Starting server at http://127.0.0.1:8123")
    uvicorn.run(app, host="127.0.0.1", port=8123)
