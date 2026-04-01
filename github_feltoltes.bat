@echo off
echo ================================================
echo   ArchMorph - GitHub feltoltes
echo   github.com/sanyi1963bp/Clotilde-and-Matilde
echo ================================================
echo.

git init
git config user.name "sanyi1963bp"
git config user.email "sanyi1963bp@gmail.com"
git branch -M main 2>nul

git add *.py
if exist README.md git add README.md
if exist .gitignore git add .gitignore

git diff --cached --quiet
if %ERRORLEVEL%==0 (
    echo Nincs uj fajl - folytatom a push-sal...
) else (
    git commit -m "ArchMorph Professional v0.5.0 - initial commit"
)

git remote remove origin 2>nul
git remote add origin https://github.com/sanyi1963bp/Clotilde-and-Matilde.git

echo.
echo Feltoltes kovetkezik - GitHub kerni fogja a jelszo / tokent...
echo.

git push -u origin main

echo.
if %ERRORLEVEL%==0 (
    echo SIKERES! Megnezhed itt:
    echo https://github.com/sanyi1963bp/Clotilde-and-Matilde
) else (
    echo Hiba! Ellenorizd hogy letezik-e a repo:
    echo https://github.com/new  ^(nev: Clotilde-and-Matilde^)
)
echo.
pause
